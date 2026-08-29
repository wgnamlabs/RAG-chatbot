# Eval Questions Schema v2.0

This benchmark is intentionally **chunker-independent**. Gold truth is anchored to exact source evidence and section paths, not to `chunk_id`, because `semantic.json` and `hierarchical.json` create different chunk boundaries.

## Dataset size

- 220 total questions
- 160 standard in-domain
- 30 hard / multi-evidence
- 30 out-of-domain (OOD)
- 60 dev / 160 test
- 27 table-lookup questions
- 30 patient-style questions
- 20 numeric-focused questions

## Core evaluation uses

The same master set can support dense embedding retrieval, BM25/hybrid/RRF, reranking, table retrieval, multi-evidence recall, answer correctness, key-fact coverage, citation correctness, faithfulness/context metrics, and OOD abstention.

## Important evaluation rule

Do **not** score a retrieved chunk as relevant merely because its `metadata.source` equals the source document. That makes large documents artificially easy. Match against `gold_evidence` / `ground_truth_sections`, using breadcrumb/section overlap and evidence-text overlap.

## Split policy

Use `split=dev` for model/configuration tuning. Use `split=test` only after the pipeline configuration is frozen.

## Record fields

`id`, `question`, `source_doc_tag`, `source_file`, `answerable`, `expected_behavior`, `question_type`, `base_question_type`, `question_style`, `benchmark_group`, `difficulty`, `requires_table`, `requires_multiple_evidence`, `gold_evidence`, `gold_evidence_count`, `ground_truth_sections`, `ground_truth_evidence_ids`, `reference_answer`, `reference_answer_type`, `key_facts`, `evaluation_targets`, `tags`, `split`, `language`.

## OOD behavior

For `answerable=false`, the expected system behavior is to abstain or explicitly state that the answer is not supported by the provided corpus. These records are suitable for OOD detection, hallucination-rate and abstention evaluation.
