# QA PATCH REPORT — eval_questions v3.1

## Kết quả

- Tổng record: 220
- DEV / TEST: 60 / 160
- Standard / hard_multi / OOD: 160 / 30 / 30
- Gold evidence units kiểm tra lại nguyên văn với 7 file clean: 220
- Question↔evidence mismatch đã sửa: 14
- Reference/key-facts record đã sửa: 32
- Gold evidence bị thay đổi: 0
- Câu wording kiểu "Theo hướng dẫn/Theo tài liệu": 0

## Các mismatch chính đã sửa

- EVAL_QD_1139_007: wording chuyển thành lỗi cần tránh khi tư vấn thai phụ.
- EVAL_QD_1139_009: hỏi tiền sử sức khỏe gia đình, không còn hỏi "gia đình hỗ trợ".
- EVAL_QD_1154_005: hỏi chẩn đoán tăng huyết áp mạn tính, đúng section 5.1.
- EVAL_QD_1154_006/007: hỏi sàng lọc/dự phòng TSG, đúng section 7.
- EVAL_QD_1359_014: "tháo que cấy", không còn nhầm thành tháo DCTC.
- EVAL_QD_1470_004: hỏi đúng nội dung sàng lọc 3 tháng cuối.
- EVAL_QD_1470_010: hỏi bú mẹ + tránh thai hậu sản, đúng evidence.
- EVAL_QD_2323_004/008/009: sửa đúng chủ đề chăm sóc thường quy, NCBSM và mẫu máu khô.
- EVAL_HARD_010: sàng lọc/dự phòng TSG.
- EVAL_HARD_011: sản giật vs HELLP.
- EVAL_HARD_023: mẫu máu khô vs SpO2.

## Table/reference fixes

Các câu table lookup hỏi nhiều giá trị đã được sửa `reference_answer` + `key_facts`,
không còn tình trạng câu hỏi hỏi 3–4 ô nhưng đáp án chuẩn chỉ chứa 1 ô.

## DEV OOD v3.1

OOD_001, OOD_002, OOD_003, OOD_004, OOD_005, OOD_006, OOD_021, OOD_022, OOD_023, OOD_024
