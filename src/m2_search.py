"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Dùng underthesea để tách từ tiếng Việt đúng ranh giới
    # "nghỉ phép" = 1 từ ghép, không phải 2 từ riêng lẻ
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text  # fallback nếu underthesea chưa cài


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # Bước 1: Lưu documents gốc để trả về kết quả sau
        self.documents = chunks

        # Bước 2: Segment mỗi chunk và tách thành danh sách token
        # Ví dụ: "nghỉ phép năm" → ["nghỉ_phép", "năm"] (sau segment)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]

        # Bước 3: Xây dựng BM25 index từ corpus tokens
        # BM25Okapi là phiên bản BM25 phổ biến nhất
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        # Bước 1: Segment và tokenize query giống như lúc index
        tokenized_query = segment_vietnamese(query).split()

        # Bước 2: Tính BM25 score cho tất cả documents
        scores = self.bm25.get_scores(tokenized_query)

        # Bước 3: Lấy top-k indices có score cao nhất
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Bước 4: Trả về SearchResult với method="bm25"
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Chỉ trả về kết quả có điểm > 0
                results.append(SearchResult(
                    text=self.documents[idx]["text"],
                    score=float(scores[idx]),
                    metadata=self.documents[idx].get("metadata", {}),
                    method="bm25"
                ))

        # Nếu không có kết quả nào > 0, vẫn trả về top_k đầu tiên
        if not results and top_indices:
            for idx in top_indices[:min(top_k, len(top_indices))]:
                results.append(SearchResult(
                    text=self.documents[idx]["text"],
                    score=float(scores[idx]),
                    metadata=self.documents[idx].get("metadata", {}),
                    method="bm25"
                ))

        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Bước 1: Tạo (hoặc tái tạo) collection trong Qdrant
        # COSINE distance phù hợp với bge-m3 embeddings
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        # Bước 2: Encode tất cả chunks thành dense vectors
        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=True)

        # Bước 3: Tạo PointStruct và upload lên Qdrant
        # Mỗi point có: id (int), vector (embedding), payload (metadata + text)
        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # Bước 1: Encode query thành vector
        query_vector = self._get_encoder().encode(query).tolist()

        # Bước 2: Tìm kiếm trong Qdrant theo cosine similarity
        try:
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )
        except Exception:
            return []

        # Bước 3: Trả về SearchResult với method="dense"
        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata=hit.payload,
                method="dense"
            )
            for hit in hits
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Dict lưu RRF score và SearchResult gốc cho mỗi document (key = text)
    rrf_scores: dict[str, dict] = {}

    # Bước 1: Tính RRF score từ mỗi ranked list
    # Công thức: score(d) = Σ 1/(k + rank_i(d)) với k=60 theo paper gốc
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            # Cộng dồn reciprocal rank từ mỗi list (rank bắt đầu từ 0)
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Bước 2: Sắp xếp theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    # Bước 3: Trả về top_k kết quả với method="hybrid" và score = RRF score
    merged = []
    for item in sorted_items[:top_k]:
        r = item["result"]
        merged.append(SearchResult(
            text=r.text,
            score=item["score"],
            metadata=r.metadata,
            method="hybrid"
        ))

    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
