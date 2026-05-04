"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # Bước 1: Tách văn bản thành các câu riêng lẻ
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]

    if not sentences:
        return []

    # Xử lý trường hợp chỉ có 1 câu
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={**metadata, "chunk_index": 0, "strategy": "semantic"}
        )]

    # Bước 2: Encode tất cả câu bằng SentenceTransformer (all-MiniLM-L6-v2 là model nhỏ, nhanh)
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # Bước 3: Hàm tính cosine similarity giữa 2 vector
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    # Bước 4: Nhóm các câu liên tiếp có cosine similarity >= threshold vào cùng 1 chunk
    # Khi similarity < threshold → tách thành chunk mới
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Similarity thấp → tạo chunk mới
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    # Đừng quên nhóm câu cuối cùng
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents = []
    children = []

    # Bước 1: Tách text thành parent chunks theo paragraph
    # Gom các paragraph cho đến khi đạt parent_size → tạo 1 parent chunk
    paragraphs = text.split("\n\n")
    p_index = 0
    current_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Nếu thêm paragraph này vào sẽ vượt parent_size và current_text không rỗng → lưu parent
        if len(current_text) + len(para) > parent_size and current_text:
            pid = f"parent_{p_index}"
            parent = Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            )
            parents.append(parent)

            # Bước 2: Tách parent vừa tạo thành các child chunks nhỏ hơn
            parent_text = current_text.strip()
            c_start = 0
            while c_start < len(parent_text):
                c_end = min(c_start + child_size, len(parent_text))
                child_text = parent_text[c_start:c_end].strip()
                if child_text:
                    child = Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child"},
                        parent_id=pid  # Mỗi child lưu parent_id để có thể lookup parent sau
                    )
                    children.append(child)
                c_start += child_size

            p_index += 1
            current_text = para + "\n\n"
        else:
            current_text += para + "\n\n"

    # Xử lý phần text còn lại chưa tạo thành parent
    if current_text.strip():
        pid = f"parent_{p_index}"
        parent = Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        )
        parents.append(parent)

        parent_text = current_text.strip()
        c_start = 0
        while c_start < len(parent_text):
            c_end = min(c_start + child_size, len(parent_text))
            child_text = parent_text[c_start:c_end].strip()
            if child_text:
                child = Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                )
                children.append(child)
            c_start += child_size

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # Bước 1: Tách text theo markdown headers (# ## ###)
    # regex: tìm các dòng bắt đầu bằng 1-3 dấu # (header level 1-3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    # Bước 2: Ghép header với content tương ứng
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Gặp header mới → lưu chunk của section trước (nếu có content)
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Đừng quên section cuối cùng
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
        ))
    elif current_header and not current_content.strip():
        # Header không có content → tạo chunk chỉ với header
        chunks.append(Chunk(
            text=current_header,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}
        ))

    # Nếu không có header nào → xử lý như 1 chunk đơn
    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure"}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}

    # Gom tất cả text từ documents
    all_texts = [d["text"] for d in documents]
    combined = "\n\n".join(all_texts) if all_texts else ""

    if not combined.strip():
        return {"basic": {}, "semantic": {}, "hierarchical": {}, "structure": {}}

    # Bước 1: Chạy từng strategy trên tất cả documents
    strategies = {}

    # Basic chunking
    basic_chunks = []
    for d in documents:
        basic_chunks.extend(chunk_basic(d["text"], metadata=d.get("metadata", {})))
    strategies["basic"] = basic_chunks

    # Semantic chunking
    semantic_chunks = []
    for d in documents:
        semantic_chunks.extend(chunk_semantic(d["text"], metadata=d.get("metadata", {})))
    strategies["semantic"] = semantic_chunks

    # Hierarchical chunking — trả về (parents, children), dùng children để so sánh
    hier_all_parents = []
    hier_all_children = []
    for d in documents:
        p, c = chunk_hierarchical(d["text"], metadata=d.get("metadata", {}))
        hier_all_parents.extend(p)
        hier_all_children.extend(c)
    strategies["hierarchical"] = hier_all_children  # So sánh dựa trên children

    # Structure-aware chunking
    struct_chunks = []
    for d in documents:
        struct_chunks.extend(chunk_structure_aware(d["text"], metadata=d.get("metadata", {})))
    strategies["structure"] = struct_chunks

    # Bước 2: Tính stats cho mỗi strategy
    def calc_stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": int(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    for name, chunks in strategies.items():
        results[name] = calc_stats(chunks)
        if name == "hierarchical":
            results[name]["num_parents"] = len(hier_all_parents)
            results[name]["num_children"] = len(hier_all_children)

    # Bước 3: In bảng so sánh đẹp
    print(f"\n{'Strategy':<15} | {'Chunks':>8} | {'Avg Len':>8} | {'Min':>6} | {'Max':>6}")
    print("-" * 55)
    for name, stats in results.items():
        n = stats.get("num_chunks", 0)
        print(f"{name:<15} | {n:>8} | {stats.get('avg_length', 0):>8} | {stats.get('min_length', 0):>6} | {stats.get('max_length', 0):>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
