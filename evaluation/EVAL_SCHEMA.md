# EVAL_SCHEMA — Master RAG Benchmark v3.1

## Mục tiêu

Bộ `eval_questions.json` này là benchmark dùng xuyên suốt pipeline:

- chunking / embedding retrieval
- dense / sparse / hybrid retrieval
- reranking
- context precision / recall
- generation correctness
- key-fact coverage
- citation
- OOD / abstention
- query rewriting

## Nguyên tắc v3

`question` phải giống câu hỏi thực tế của người dùng hoặc câu hỏi lâm sàng tự nhiên.

Không dùng wording máy móc kiểu:

- “Theo hướng dẫn...”
- “Theo tài liệu...”
- “Nội dung chính của mục...”
- “Theo Bảng X...”

trừ khi use case thực sự là tra cứu cấu trúc văn bản.

Ground truth **không thay đổi theo cách viết câu hỏi**. `gold_evidence` vẫn là đoạn nguyên văn từ 7 file clean.

## Các trường

### Nhận dạng và query

- `id`: ID ổn định, duy nhất.
- `question`: raw query tiếng Việt đưa vào pipeline.
- `language`: `vi`.

### Nguồn và khả năng trả lời

- `source_doc_tag`: tag tài liệu; `null` với OOD.
- `source_file`: tên file clean; `null` với OOD.
- `answerable`: corpus có đủ evidence hay không.
- `expected_behavior`: `answer_from_corpus` hoặc `abstain_or_state_not_in_corpus`.

### Phân loại query

- `question_type`: loại năng lực cần truy xuất, độc lập với cách diễn đạt.
  Ví dụ: `definition`, `fact`, `numeric`, `risk_factor`, `diagnosis`,
  `symptom`, `procedure`, `recommendation`, `counseling`,
  `table_lookup`, `comparison`, `multi_evidence`, `out_of_domain`.
- `base_question_type`: loại nền được giữ để tương thích phân tích cũ.
- `question_style`:
  - `patient_style`: câu hỏi/scenario gần cách bệnh nhân hoặc người chăm sóc hỏi.
  - `natural`: câu hỏi trung tính tự nhiên.
  - `clinical`: câu hỏi chuyên môn hoặc staff-facing.
- `difficulty`: `easy`, `medium`, `hard`.
- `benchmark_group`: `standard`, `hard_multi`, `ood`.

### Query features

`query_features` gồm:

- `colloquial`: wording hội thoại/đời thường.
- `short_query`: query ngắn <= 10 từ.
- `contains_typo`: có chủ động đưa lỗi chính tả vào benchmark hay không.
- `implicit_medical_term`: dùng cách gọi đời thường thay cho thuật ngữ/heading nguồn.

Field này hữu ích để đánh giá query rewriting theo từng slice.

### Ground truth

- `requires_table`: evidence nằm trong Markdown table.
- `requires_multiple_evidence`: cần nhiều evidence units.
- `gold_evidence`: danh sách evidence độc lập chunker:
  - `evidence_id`
  - `source_doc_tag`
  - `source_file`
  - `section_path`
  - `heading`
  - `start_line`
  - `end_line`
  - `evidence_text`
- `gold_evidence_count`
- `ground_truth_sections`
- `ground_truth_evidence_ids`

Không dùng `chunk_id` làm gold ground truth vì Semantic và Hierarchical tạo chunk khác nhau.

### Generation reference

- `reference_answer`: đáp án chuẩn bám nguồn.
- `reference_answer_type`: `extractive`, `extractive_composite`, `abstention`.
- `key_facts`: các ý nguyên tử cần có trong câu trả lời đúng.

### Evaluation metadata

- `evaluation_targets`
- `tags`
- `split`: `dev` hoặc `test`.

## DEV / TEST

- DEV: 60 câu.
- TEST: 160 câu.
- 30 câu hard/multi-evidence.
- 30 câu OOD.

Dùng DEV để chọn chunker, embedding, top-k, hybrid weights, reranker.
TEST chỉ dùng sau khi đã chốt cấu hình.

## Lưu ý với embedding evaluation

Metric final phải match retrieved chunks với `gold_evidence`, không chỉ kiểm tra `source_file`.

Các metric nên dùng:

- Evidence Recall@K
- Hit@K
- Evidence Complete@K
- MRR
- nDCG@K
- OOD AUROC

## Lưu ý với query rewriting

Luôn đưa `question` nguyên bản vào benchmark raw retrieval trước.
Sau đó mới chạy một experiment riêng:

`question -> query rewriter -> retrieval`

để đo mức cải thiện do rewriting.


## QA revision v3.1

- Sửa các câu có wording không khớp `gold_evidence` sau lần rewrite v3.
- Sửa `reference_answer` và `key_facts` cho các câu bảng hỏi nhiều ô/hàng.
- Không thay đổi `gold_evidence` của 220 câu.
- DEV vẫn 60 câu, TEST vẫn 160 câu.
- DEV OOD vẫn 10 câu nhưng có thêm các câu health-adjacent/dynamic để tránh OOD quá dễ.
- Tag `qa_fixed_v3_1` đánh dấu mọi record thuộc bản QA final này.
