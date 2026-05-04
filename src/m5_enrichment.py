"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """Lấy OpenAI client (lazy init)."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Giữ lại thông tin quan trọng nhất."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback extractive: lấy 2 câu đầu (không cần API)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) >= 2:
        return ". ".join(sentences[:2]) + "."
    return text[:200] + "..." if len(text) > 200 else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Ví dụ: user hỏi "nghỉ phép bao nhiêu ngày?" nhưng doc viết "12 ngày làm việc mỗi năm"
    HyQA index thêm câu hỏi "Nhân viên được nghỉ bao nhiêu ngày?" → tăng recall
    """
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            questions = raw.split("\n")
            # Làm sạch: bỏ số thứ tự, dấu gạch đầu dòng
            questions = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
            return questions[:n_questions]
        except Exception:
            pass

    # Fallback: tạo câu hỏi đơn giản từ text
    words = text.split()[:10]
    snippet = " ".join(words)
    fallback_qs = [
        f"Thông tin về {snippet[:30]}... là gì?",
        f"Quy định liên quan đến nội dung này là gì?",
        f"Điều kiện áp dụng trong trường hợp này là gì?",
    ]
    return fallback_qs[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Kết quả: "Trích từ [tài liệu], phần về [chủ đề].\n\n[original text]"
    """
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            doc_hint = f"Tài liệu: {document_title}\n\n" if document_title else ""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất, không giải thích thêm."},
                    {"role": "user", "content": f"{doc_hint}Đoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context_sentence = resp.choices[0].message.content.strip()
            # Ghép context + original text (original phải được giữ nguyên!)
            return f"{context_sentence}\n\n{text}"
        except Exception:
            pass

    # Fallback: thêm document title đơn giản nếu có
    if document_title:
        return f"Trích từ tài liệu: {document_title}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    Metadata này gắn vào chunk → enable rich filtering khi search.
    """
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ với đúng format: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}. Chỉ trả về JSON, không giải thích.'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            # Làm sạch response (đôi khi có ```json wrapper)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except Exception:
            pass

    # Fallback: heuristic metadata dựa trên keyword
    text_lower = text.lower()
    category = "policy"
    if any(w in text_lower for w in ["mật khẩu", "vpn", "bảo mật", "it", "password"]):
        category = "it"
    elif any(w in text_lower for w in ["lương", "thưởng", "tài chính", "budget"]):
        category = "finance"
    elif any(w in text_lower for w in ["nghỉ phép", "tuyển dụng", "nhân viên", "thử việc"]):
        category = "hr"

    return {
        "topic": text[:50].strip(),
        "entities": [],
        "category": category,
        "language": "vi"
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        text = chunk["text"]
        meta = chunk.get("metadata", {})
        doc_title = meta.get("source", "")

        # Technique 1: Summary (nếu được yêu cầu)
        summary = ""
        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(text)

        # Technique 2: Hypothesis Questions (bridge vocabulary gap)
        questions = []
        if "hyqa" in methods or "full" in methods:
            questions = generate_hypothesis_questions(text)

        # Technique 3: Contextual Prepend (giúp LLM hiểu chunk nằm ở đâu)
        enriched_text = text  # default: giữ nguyên
        if "contextual" in methods or "full" in methods:
            enriched_text = contextual_prepend(text, doc_title)

        # Technique 4: Auto Metadata (enable rich filtering)
        auto_meta = {}
        if "metadata" in methods or "full" in methods:
            auto_meta = extract_metadata(text)

        enriched.append(EnrichedChunk(
            original_text=text,  # Quan trọng: giữ nguyên text gốc
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},  # Merge metadata gốc + extracted
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
