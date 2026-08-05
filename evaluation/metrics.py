from typing import List, Dict, Any


def get_doc_id(doc_metadata: Dict[str, Any]) -> str:
    """Trích ID tài liệu từ metadata chunk — dùng trường 'source'."""
    return str(doc_metadata.get("source", doc_metadata.get("file_name", "")))


def recall_at_k(
    retrieved_metadata: List[Dict[str, Any]],
    ground_truth_sources: List[str],
    k: int,
) -> float:
    """Tỷ lệ nguồn liên quan được tìm thấy trong top-k.

    Với bộ câu hỏi này mỗi câu chỉ có ≤1 ground truth source nên
    recall@k ∈ {0, 1}.  Nếu ground_truth_sources rỗng (câu hỏi
    ngoài miền) trả 0 thay vì chia cho 0.
    """
    if not ground_truth_sources:
        return 0.0
    retrieved_k = retrieved_metadata[:k]
    retrieved_sources = {get_doc_id(m) for m in retrieved_k}
    relevant_retrieved = retrieved_sources.intersection(set(ground_truth_sources))
    return len(relevant_retrieved) / len(ground_truth_sources)


def precision_at_k(
    retrieved_metadata: List[Dict[str, Any]],
    ground_truth_sources: List[str],
    k: int,
) -> float:
    """Tỷ lệ chunk trong top-k đến từ đúng nguồn liên quan.

    BUG CŨ: dùng set(retrieved_sources) → nếu top-5 là [đúng, đúng, đúng,
    sai, sai], set chỉ còn {đúng} → len=1 → precision=1/5=0.2 thay vì 0.6.
    FIX: đếm TỪNG CHUNK một (hits), không collapse về set.
    """
    retrieved_k = retrieved_metadata[:k]
    if not retrieved_k:
        return 0.0
    gt_set = set(ground_truth_sources)
    hits = sum(1 for m in retrieved_k if get_doc_id(m) in gt_set)
    return hits / len(retrieved_k)


def mean_reciprocal_rank(
    retrieved_metadata: List[Dict[str, Any]],
    ground_truth_sources: List[str],
) -> float:
    """Reciprocal rank của chunk liên quan đầu tiên trong danh sách trả về."""
    gt_set = set(ground_truth_sources)
    for i, meta in enumerate(retrieved_metadata):
        if get_doc_id(meta) in gt_set:
            return 1.0 / (i + 1)
    return 0.0
