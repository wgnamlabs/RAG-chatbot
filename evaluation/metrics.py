"""Metrics cho master eval_questions.json v2.

Khác version cũ:
- KHÔNG coi "đúng file" là relevant.
- Ground truth là gold_evidence độc lập với chunker.
- Mỗi gold evidence được map sang chunk bằng đoạn token liên tiếp xuất hiện
  trong chunk. Chỉ xét chunk cùng source_file.
- Với câu multi-evidence, Recall@k là tỷ lệ evidence units được tìm thấy.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class EvidenceMatchConfig:
    # Một chunk được coi là relevant với evidence khi chứa tối thiểu 30%
    # token của evidence theo một đoạn liên tiếp.
    min_coverage: float = 0.30

    # Đồng thời phải khớp ít nhất 8 token liên tiếp để tránh false positive
    # trên các cụm y khoa/ngôn ngữ chung.
    min_contiguous_tokens: int = 8


DEFAULT_EVIDENCE_MATCH = EvidenceMatchConfig()


def normalize_nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def get_doc_id(doc_metadata: Dict[str, Any]) -> str:
    raw = doc_metadata.get("source", doc_metadata.get("file_name", ""))
    return normalize_nfc(raw)


def strip_chunk_breadcrumb(text: str) -> str:
    """Bỏ dòng [breadcrumb] được prepend vào text để matching evidence sạch hơn."""
    lines = str(text or "").splitlines()
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        lines = lines[1:]
    return "\n".join(lines)


def tokenize_for_evidence(text: str) -> List[str]:
    """Tokenizer nhẹ, deterministic, không phụ thuộc model embedding."""
    text = normalize_nfc(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    # Bỏ markup Markdown nhưng giữ nội dung bảng/số liệu.
    text = re.sub(r"[#*_`>|]+", " ", text)
    return re.findall(
        r"\w+(?:[.,]\d+)?|>=|<=|≥|≤|>|<|%",
        text,
        flags=re.UNICODE,
    )


def contiguous_evidence_overlap(
    evidence_text: str,
    chunk_text: str,
) -> Tuple[float, int]:
    """Trả (coverage, matched_tokens).

    coverage = số token của đoạn khớp liên tiếp dài nhất / số token evidence.

    Vì chunks được sinh trực tiếp từ cùng Markdown sạch, contiguous matching
    phù hợp hơn bag-of-words: tránh đánh dấu nhầm chunk chỉ vì dùng cùng thuật ngữ.
    """
    evidence_tokens = tokenize_for_evidence(evidence_text)
    chunk_tokens = tokenize_for_evidence(strip_chunk_breadcrumb(chunk_text))

    if not evidence_tokens or not chunk_tokens:
        return 0.0, 0

    match = difflib.SequenceMatcher(
        None,
        evidence_tokens,
        chunk_tokens,
        autojunk=False,
    ).find_longest_match(
        0,
        len(evidence_tokens),
        0,
        len(chunk_tokens),
    )

    return match.size / len(evidence_tokens), match.size


def is_relevant_overlap(
    coverage: float,
    matched_tokens: int,
    evidence_token_count: int,
    config: EvidenceMatchConfig = DEFAULT_EVIDENCE_MATCH,
) -> bool:
    # Với evidence rất ngắn, không yêu cầu nhiều token hơn chính evidence.
    min_tokens = min(config.min_contiguous_tokens, evidence_token_count)
    return coverage >= config.min_coverage and matched_tokens >= min_tokens


def build_gold_relevance(
    qa_item: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    config: EvidenceMatchConfig = DEFAULT_EVIDENCE_MATCH,
) -> Dict[str, Any]:
    """Map gold evidence -> relevant chunk indices.

    Hàm này chỉ dùng ground truth để CHẤM ĐIỂM. Nó không tác động embedding,
    similarity hay ranking, nên không gây retrieval leakage.
    """
    if not qa_item.get("answerable", False):
        return {
            "evidence_to_chunks": {},
            "chunk_scores": {},
            "unmapped_evidence": [],
            "best_candidates": {},
        }

    source_file = normalize_nfc(qa_item.get("source_file"))
    candidate_indices = [
        i
        for i, chunk in enumerate(chunks)
        if get_doc_id(chunk.get("metadata", {})) == source_file
    ]

    evidence_to_chunks: Dict[str, Set[int]] = {}
    chunk_scores: Dict[int, float] = {}
    unmapped: List[str] = []
    best_candidates: Dict[str, Dict[str, Any]] = {}

    for evidence in qa_item.get("gold_evidence", []):
        evidence_id = evidence["evidence_id"]
        evidence_text = evidence["evidence_text"]
        evidence_token_count = len(tokenize_for_evidence(evidence_text))

        relevant_indices: Set[int] = set()
        best_score = -1.0
        best_match_tokens = 0
        best_idx = None

        for idx in candidate_indices:
            chunk = chunks[idx]
            coverage, matched_tokens = contiguous_evidence_overlap(
                evidence_text,
                chunk.get("text", ""),
            )

            if coverage > best_score:
                best_score = coverage
                best_match_tokens = matched_tokens
                best_idx = idx

            if is_relevant_overlap(
                coverage,
                matched_tokens,
                evidence_token_count,
                config,
            ):
                relevant_indices.add(idx)
                chunk_scores[idx] = max(chunk_scores.get(idx, 0.0), coverage)

        evidence_to_chunks[evidence_id] = relevant_indices

        if not relevant_indices:
            unmapped.append(evidence_id)

        if best_idx is not None:
            best_chunk = chunks[best_idx]
            best_candidates[evidence_id] = {
                "chunk_index_in_file": best_idx,
                "chunk_id": best_chunk.get("metadata", {}).get("chunk_id"),
                "coverage": round(float(best_score), 6),
                "matched_tokens": int(best_match_tokens),
                "evidence_tokens": int(evidence_token_count),
            }

    return {
        "evidence_to_chunks": evidence_to_chunks,
        "chunk_scores": chunk_scores,
        "unmapped_evidence": unmapped,
        "best_candidates": best_candidates,
    }


def relevant_chunk_indices(gold_map: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    for indices in gold_map.get("evidence_to_chunks", {}).values():
        out.update(indices)
    return out


def evidence_recall_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    evidence_map = gold_map.get("evidence_to_chunks", {})
    if not evidence_map:
        return 0.0

    top_k = set(ranking[:k])
    hit_evidence = sum(
        1 for relevant_indices in evidence_map.values()
        if top_k.intersection(relevant_indices)
    )
    return hit_evidence / len(evidence_map)


def recall_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    """Alias: Recall@k trong benchmark mới = evidence recall@k."""
    return evidence_recall_at_k(ranking, gold_map, k)


def hit_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    return float(evidence_recall_at_k(ranking, gold_map, k) > 0.0)


def complete_evidence_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    evidence_map = gold_map.get("evidence_to_chunks", {})
    if not evidence_map:
        return 0.0
    return float(evidence_recall_at_k(ranking, gold_map, k) >= 1.0)


def precision_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    top_k = list(ranking[:k])
    if not top_k:
        return 0.0

    relevant = relevant_chunk_indices(gold_map)
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / len(top_k)


def reciprocal_rank(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
) -> float:
    relevant = relevant_chunk_indices(gold_map)
    for rank, idx in enumerate(ranking, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[Sequence[int]],
    gold_maps: Iterable[Dict[str, Any]],
) -> float:
    vals = [
        reciprocal_rank(ranking, gold_map)
        for ranking, gold_map in zip(rankings, gold_maps)
    ]
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(
    ranking: Sequence[int],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    """nDCG với graded relevance = evidence coverage của chunk."""
    scores = gold_map.get("chunk_scores", {})
    if not scores or k <= 0:
        return 0.0

    ranked_rels = [float(scores.get(idx, 0.0)) for idx in ranking[:k]]

    def dcg(rels: Sequence[float]) -> float:
        return sum(
            rel / math.log2(rank + 2)
            for rank, rel in enumerate(rels)
        )

    actual = dcg(ranked_rels)
    ideal_rels = sorted(scores.values(), reverse=True)[:k]
    ideal = dcg(ideal_rels)

    return actual / ideal if ideal > 0 else 0.0


def ood_auroc(
    in_domain_top1_sims: Sequence[float],
    ood_top1_sims: Sequence[float],
) -> float:
    """AUROC cho OOD detection.

    OOD được coi là positive và similarity thấp hơn là tín hiệu OOD.
    Tính trực tiếp P(sim_OOD < sim_ID), ties tính 0.5.
    """
    if not in_domain_top1_sims or not ood_top1_sims:
        return float("nan")

    wins = 0.0
    total = 0

    for ood_sim in ood_top1_sims:
        for id_sim in in_domain_top1_sims:
            total += 1
            if ood_sim < id_sim:
                wins += 1.0
            elif ood_sim == id_sim:
                wins += 0.5

    return wins / total if total else float("nan")
