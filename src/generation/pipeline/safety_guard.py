"""
safety_guard.py — Lớp an toàn cứng (hard fallback) cho các triệu chứng nguy hiểm
và các yêu cầu adversarial (ép kê đơn / ép chẩn đoán / lách khuyến cáo).

Bối cảnh 1 (danger guard): eval cho thấy case "Thai máy yếu dần, không thấy
máy nữa" bị lỗi generation thật — model vừa nói "không tìm thấy thông tin"
vừa tiếp tục suy đoán, vì retrieval không khớp được từ dân dã "thai máy" với
thuật ngữ y khoa trong tài liệu ("cử động thai"). Đây là rủi ro an toàn không
nên phụ thuộc hoàn toàn vào retrieval + generation để tự xử lý đúng.

Bối cảnh 2 (adversarial guard): Các test case cho thấy câu hỏi
adversarial trong bộ eval dễ bị FAIL — model không có tín hiệu từ chối rõ ràng
khi bị ép:
  - yêu cầu liều thuốc cụ thể ("liều lượng bao nhiêu mg")
  - ép xác nhận/khẳng định chẩn đoán thay bác sĩ ("xác nhận giúp tôi luôn")
  - cố lách qua khuyến cáo đi khám để lấy liều dùng cụ thể
Cũng như danger guard, đây là hành vi không nên phụ thuộc vào generation tự
xử lý đúng — cần một lớp chặn cứng, độc lập, dựa trên chính CÂU HỎI.

Module này KHÔNG thay thế retrieval/generation — nó là lớp kiểm tra cuối
cùng, độc lập, chạy sau khi có answer (hoặc trước, để override), dựa trên
keyword-matching trên chính CÂU HỎI (không phải câu trả lời), nên không phụ
thuộc vào việc retrieval có tìm đúng chunk hay không, hay việc generation có
tự "nhớ" nói câu từ chối hay không.

Cách tích hợp vào pipeline (cần xem main.py để gắn đúng chỗ — xem docstring
cuối file):
    from evaluation.safety_guard import apply_safety_guard
    ...
    pipeline_output.answer = apply_safety_guard(query, pipeline_output.answer)
"""

from __future__ import annotations

import re

# ── Từ khóa nguy hiểm — bao gồm cả thuật ngữ y khoa lẫn cách nói dân dã ──────
# Mỗi nhóm là các cách diễn đạt khác nhau của CÙNG một tình huống nguy hiểm,
# để bắt được câu hỏi dù người dùng dùng từ ngữ nào.
DANGER_PATTERNS: dict[str, list[str]] = {
    "giam_cu_dong_thai": [
        "thai máy yếu", "không thấy máy", "không thấy đạp", "không đạp nữa",
        "hết cử động", "không cử động", "thai không máy", "bé không đạp",
        "im re không đạp",
    ],
    "ra_mau_bat_thuong": [
        "ra máu đỏ tươi", "ra máu nhiều", "chảy máu âm đạo", "ra huyết nhiều",
        "băng huyết", "ra máu tươi", "máu ra nhiều", "ướt hết băng vệ sinh",
    ],
    "san_giat_tien_san_giat": [
        "co giật", "giật toàn thân", "mắt mờ", "hoa mắt", "huyết áp cao",
        "phù toàn thân", "nhức đầu dữ dội",
    ],
    "vo_oi_bat_thuong": [
        "vỡ ối", "ối vỡ", "nước ối xanh", "nước ối có mùi", "rỉ ối",
    ],
    "nhiem_trung_sot_cao": [
        "sốt cao", "khó thở", "đau ngực", "ớn lạnh dữ dội",
    ],
    "dau_bung_du_doi": [
        "đau bụng dữ dội", "đau quặn bụng", "đau bụng không chịu nổi",
        "co thắt liên tục", "đau bụng từng cơn mạnh",
    ],
}

# Câu trả lời coi như ĐÃ có đủ mức khẩn cấp nếu chứa 1 trong các cụm này —
# khi đó không cần chèn thêm dòng an toàn (tránh lặp ý).
_URGENCY_ALREADY_PRESENT = [
    "cấp cứu", "115", "đến bệnh viện ngay", "đến viện ngay",
    "không nên trì hoãn", "không nên chờ", "ngay lập tức",
]

_SAFETY_LINE = (
    "\n\n⚠️ Lưu ý quan trọng: những dấu hiệu chị vừa mô tả có thể liên quan "
    "đến một tình huống cần được xử lý sớm. Để đảm bảo an toàn cho chị và "
    "em bé, chị nên đến cơ sở y tế gần nhất hoặc gọi cấp cứu 115 ngay, "
    "không nên chờ đợi hay tự theo dõi thêm ở nhà."
)

_INCONSISTENT_REFUSAL_MARKER = "không tìm thấy thông tin"
_MIN_TRAILING_FOR_INCONSISTENT = 60


# ── Adversarial patterns — ép kê đơn / ép chẩn đoán / lách khuyến cáo ────────
# Lấy trực tiếp từ các case FAIL thật, viết rộng hơn
# một chút để bắt được các biến thể gần giống chứ không chỉ đúng câu đó.
ADVERSARIAL_PATTERNS: dict[str, list[str]] = {
    "yeu_cau_lieu_thuoc_cu_the": [
        "liều lượng bao nhiêu mg", "liều bao nhiêu mg", "bao nhiêu mg thì",
        "mg là an toàn", "liều nào là an toàn", "liều dùng chính xác",
        "liều chính xác", "uống thuốc gì và liều", "cho tôi biết chính xác",
        "kê đơn giúp tôi", "kê cho tôi", "cho tôi xin liều",
        # Pattern mới — case FAIL (2026-08-22)
        "tên và liều thuốc", "liều thuốc hạ huyết áp", "liều magie sulfat",
        "magie sulfat để tự tiêm", "oxytocin để tự tiêm", "oxytocin để tự",
        "giục sinh tại nhà", "tự tiêm tại nhà", "liều để tự tiêm",
        "thuốc gì để tự xử lý", "tự xử lý ở nhà",
    ],
    "ep_xac_nhan_chan_doan": [
        "xác nhận giúp tôi", "xác nhận giúp", "khẳng định giúp tôi",
        "chốt giúp tôi", "chốt luôn giúp tôi", "tôi chắc chắn",
        "chắc chắn là", "chắc chắn 90%", "xác nhận luôn đi",
        "kết luận giúp tôi", "nói chắc giúp tôi",
        # Pattern mới — case FAIL
        "chắc chắn 95%", "khỏi cần đi khám nữa",
    ],
    "lach_khuyen_cao": [
        "bỏ qua khuyến cáo", "bỏ qua mấy cái khuyến cáo", "bỏ qua lời khuyên",
        "không cần đi khám", "khỏi cần khám", "đừng bảo tôi đi khám",
        "đừng khuyên tôi đi khám", "không cần khuyên tôi đi khám",
        # Pattern mới — case FAIL (2026-08-22)
        "đừng bảo tôi đi cấp cứu", "đừng nhắc tôi đi khám",
        "không muốn đi viện", "không muốn đi bệnh viện",
        "tự cầm máu tại nhà", "cách tự cầm máu",
    ],
}

# Câu trả lời coi như ĐÃ có tín hiệu từ chối kê đơn/chẩn đoán rõ ràng rồi —
# khi đó không cần ghi đè (tránh mất nội dung tư vấn hợp lệ đã có).
_REFUSAL_ALREADY_PRESENT = [
    "không thể kê đơn", "không kê đơn", "không thể chẩn đoán xác định",
    "không thể chẩn đoán thay", "không thể xác nhận chẩn đoán",
    "không thể khẳng định chẩn đoán", "cần bác sĩ khám trực tiếp",
    "bác sĩ trực tiếp thăm khám mới có thể",
]

_ADVERSARIAL_REFUSAL_LINE = (
    "Mình không thể kê đơn thuốc, chỉ định liều lượng cụ thể, hay chẩn đoán "
    "xác định/khẳng định chắc chắn tình trạng bệnh thay bác sĩ — những việc "
    "này cần bác sĩ khám trực tiếp, kèm xét nghiệm hoặc siêu âm nếu cần, mới "
    "đảm bảo chính xác và an toàn cho chị và em bé (liều dùng còn phụ thuộc "
    "tuần thai, cân nặng và tình trạng sức khỏe cụ thể của từng người). Chị "
    "vui lòng liên hệ bác sĩ đang theo dõi thai kỳ hoặc đến cơ sở y tế để "
    "được thăm khám, kê đơn và chẩn đoán chính thức — mình không thể kê đơn "
    "hay xác nhận chẩn đoán thay việc đó."
)

_ADVERSARIAL_REFUSAL_TEXT: dict[str, str] = {
    "yeu_cau_lieu_thuoc_cu_the": _ADVERSARIAL_REFUSAL_LINE,
    "ep_xac_nhan_chan_doan": _ADVERSARIAL_REFUSAL_LINE,
    "lach_khuyen_cao": _ADVERSARIAL_REFUSAL_LINE,
}


def detect_adversarial_signal(question: str) -> str | None:
    """Trả về tên nhóm adversarial đầu tiên khớp với câu hỏi, hoặc None."""
    q_lower = question.lower()
    for group, patterns in ADVERSARIAL_PATTERNS.items():
        for p in patterns:
            if p in q_lower:
                return group
    return None


def _has_refusal_language(answer: str) -> bool:
    a_lower = answer.lower()
    return any(kw in a_lower for kw in _REFUSAL_ALREADY_PRESENT)


def apply_adversarial_guard(question: str, answer: str) -> str:
    """
    Lớp chặn cứng cho câu hỏi ép kê đơn/chẩn đoán/lách khuyến cáo.

    Logic:
      1. Câu hỏi không khớp pattern adversarial nào → trả answer nguyên vẹn.
      2. Có khớp NHƯNG answer đã tự có tín hiệu từ chối rõ ràng rồi (model
         tự xử lý đúng) → giữ nguyên, không ghi đè để không mất nội dung.
      3. Có khớp và answer CHƯA có tín hiệu từ chối rõ ràng → THAY THẾ hoàn
         toàn bằng câu từ chối cố định tương ứng với nhóm phát hiện được.
         Ghi đè hoàn toàn (không chỉ chèn thêm) vì mục tiêu là đảm bảo có
         một câu từ chối rõ ràng, không lẫn với nội dung có thể vẫn chứa
         liều lượng/khẳng định chẩn đoán ở phần trước.
    """
    adversarial = detect_adversarial_signal(question)
    if adversarial is None:
        return answer
    if _has_refusal_language(answer):
        return answer
    return _ADVERSARIAL_REFUSAL_TEXT[adversarial]


def detect_danger_signal(question: str) -> str | None:
    """Trả về tên nhóm nguy hiểm đầu tiên khớp với câu hỏi, hoặc None."""
    q_lower = question.lower()
    for group, patterns in DANGER_PATTERNS.items():
        for p in patterns:
            if p in q_lower:
                return group
    return None


def _has_urgency_language(answer: str) -> bool:
    a_lower = answer.lower()
    return any(kw in a_lower for kw in _URGENCY_ALREADY_PRESENT)


def _is_inconsistent_refusal(answer: str) -> bool:
    """Vừa từ chối vừa tiếp tục sinh nội dung dài — bug đã gặp thực tế."""
    idx = answer.lower().find(_INCONSISTENT_REFUSAL_MARKER)
    if idx == -1:
        return False
    trailing = answer[idx + len(_INCONSISTENT_REFUSAL_MARKER):].strip()
    return len(trailing) > _MIN_TRAILING_FOR_INCONSISTENT


def apply_safety_guard(question: str, answer: str) -> str:
    """
    Lớp an toàn cuối cùng, gọi SAU khi pipeline sinh xong answer.

    Chạy 2 guard độc lập, theo thứ tự:

    (A) Adversarial guard — ép kê đơn / ép chẩn đoán / lách khuyến cáo:
      1. Câu hỏi không khớp pattern adversarial nào → bỏ qua bước này.
      2. Có khớp nhưng answer đã tự từ chối đúng rồi → giữ nguyên.
      3. Có khớp và answer chưa từ chối rõ ràng → THAY THẾ hoàn toàn bằng
         câu từ chối cố định (xem apply_adversarial_guard).

    (B) Danger guard — triệu chứng nguy hiểm cần khám sớm, chạy TIẾP trên
        kết quả sau bước (A):
      1. Câu hỏi không chứa dấu hiệu nguy hiểm nào → trả answer nguyên vẹn.
      2. Có dấu hiệu nguy hiểm NHƯNG answer đang ở trạng thái tự mâu thuẫn
         (vừa từ chối vừa suy đoán) → THAY THẾ hoàn toàn bằng câu trả lời an
         toàn cố định, không cố giữ lại phần nội dung mập mờ đó.
      3. Có dấu hiệu nguy hiểm, answer bình thường (không mâu thuẫn) NHƯNG
         chưa có ngôn ngữ khẩn cấp rõ ràng → CHÈN THÊM dòng khuyến cáo an
         toàn vào cuối, giữ nguyên nội dung y khoa đã có.
      4. Có dấu hiệu nguy hiểm và answer đã có ngôn ngữ khẩn cấp rõ ràng rồi
         → trả nguyên vẹn, không chèn trùng lặp.

    Hai guard độc lập nên vẫn hoạt động đúng nếu 1 câu hỏi hiếm gặp khớp cả
    2 nhóm pattern — ví dụ vừa hỏi liều thuốc cụ thể vừa mô tả triệu chứng
    nguy hiểm: answer sẽ bị adversarial guard thay bằng câu từ chối kê đơn,
    sau đó danger guard kiểm tra tiếp trên câu đó và chèn thêm dòng khuyến
    cáo cấp cứu nếu chưa có (vì _ADVERSARIAL_REFUSAL_LINE không tự chứa
    ngôn ngữ khẩn cấp dạng "115"/"cấp cứu").
    """
    answer = apply_adversarial_guard(question, answer)

    danger = detect_danger_signal(question)
    if danger is None:
        return answer

    if _is_inconsistent_refusal(answer):
        return (
            "Dựa trên mô tả của chị, đây có thể là một dấu hiệu cần được "
            "kiểm tra sớm mà tài liệu hiện có của hệ thống chưa mô tả đủ chi "
            "tiết để tư vấn cụ thể." + _SAFETY_LINE
        )

    if _has_urgency_language(answer):
        return answer

    return answer + _SAFETY_LINE


# ── Đã tích hợp vào pipeline ─────────────────────────────────────────────────
# File này đặt tại src/generation/pipeline/safety_guard.py và được gọi trực
# tiếp từ run_pipeline() trong main.py, ngay sau bước generate (và ở cả 2
# nhánh early-return: không có context / lỗi build_prompt), để đảm bảo MỌI
# đường ra khỏi pipeline đều đi qua lớp an toàn này — không phụ thuộc vào
# việc generation có thành công hay không.