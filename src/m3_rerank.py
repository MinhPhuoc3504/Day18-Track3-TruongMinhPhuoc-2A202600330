"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load cross-encoder model (lazy loading để tiết kiệm thời gian khởi động)."""
        if self._model is None:
            # Thử dùng sentence_transformers CrossEncoder (dễ cài, tương thích tốt)
            # bge-reranker-v2-m3 hỗ trợ tiếng Việt vì được train trên đa ngôn ngữ
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                # Fallback: dùng model nhỏ hơn nếu không tải được bge-reranker-v2-m3
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        # Bước 1: Load model (lần đầu sẽ chậm do tải model)
        model = self._load_model()

        # Bước 2: Tạo pairs (query, document) cho cross-encoder
        # Cross-encoder nhìn cả query + doc cùng lúc → score chính xác hơn bi-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Bước 3: Dự đoán relevance score cho từng pair
        # CrossEncoder.predict() trả về array của scores
        scores = model.predict(pairs)

        # Bước 4: Kết hợp score với document gốc
        scored_docs = list(zip(scores, documents))

        # Bước 5: Sắp xếp theo rerank_score giảm dần (doc liên quan nhất lên đầu)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Bước 6: Trả về top_k RerankResult với thông tin đầy đủ
        results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i  # rank bắt đầu từ 0
            ))

        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank dùng flashrank — nhanh hơn cross-encoder nhiều lần."""
        if not documents:
            return []

        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()

            # Tạo passages theo format của flashrank
            passages = [{"text": d["text"], "id": i} for i, d in enumerate(documents)]
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)

            reranked = []
            for i, r in enumerate(results[:top_k]):
                orig_idx = r.get("id", 0)
                orig_doc = documents[orig_idx] if orig_idx < len(documents) else documents[0]
                reranked.append(RerankResult(
                    text=r.get("text", orig_doc["text"]),
                    original_score=orig_doc.get("score", 0.0),
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=orig_doc.get("metadata", {}),
                    rank=i
                ))
            return reranked
        except Exception:
            # Fallback về CrossEncoderReranker nếu flashrank không hoạt động
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []

    # Warm-up run (lần đầu load model chậm, không tính vào benchmark)
    try:
        reranker.rerank(query, documents)
    except Exception:
        pass

    # Đo thời gian thực tế cho n_runs lần chạy
    for _ in range(n_runs):
        start = time.perf_counter()
        try:
            reranker.rerank(query, documents)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - start) * 1000  # chuyển sang milliseconds
        times.append(elapsed_ms)

    if not times:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times)
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
