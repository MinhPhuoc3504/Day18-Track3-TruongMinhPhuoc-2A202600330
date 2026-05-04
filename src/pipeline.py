"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4+M5."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

# Latency tracking cho bonus points
_latency = {"chunk": 0, "enrich": 0, "index": 0, "search": 0, "rerank": 0, "generate": 0}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1) — Hierarchical chunking
    print("\n[1/4] Chunking documents (M1: Hierarchical)...")
    t0 = time.perf_counter()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id}
            })
    _latency["chunk"] = (time.perf_counter() - t0) * 1000
    print(f"  {len(all_chunks)} child chunks from {len(docs)} documents ({_latency['chunk']:.0f}ms)")

    # Step 2: Enrichment (M5) — Contextual prepend + HyQA + metadata
    print("\n[2/4] Enriching chunks (M5: Contextual + HyQA + Metadata)...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    _latency["enrich"] = (time.perf_counter() - t0) * 1000
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks ({_latency['enrich']:.0f}ms)")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2) — BM25 + Dense Qdrant
    print("\n[3/4] Indexing (M2: BM25 + Dense Qdrant)...")
    t0 = time.perf_counter()
    search = HybridSearch()
    search.index(all_chunks)
    _latency["index"] = (time.perf_counter() - t0) * 1000
    print(f"  Indexed {len(all_chunks)} chunks ({_latency['index']:.0f}ms)")

    # Step 4: Load Reranker (M3) — Cross-encoder bge-reranker-v2-m3
    print("\n[4/4] Loading reranker (M3: bge-reranker-v2-m3)...")
    reranker = CrossEncoderReranker()

    return search, reranker


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate answer với LLM dựa trên retrieved contexts."""
    if not OPENAI_API_KEY:
        return contexts[0] if contexts else "Không tìm thấy thông tin."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        context_str = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý hỗ trợ nhân sự. Trả lời câu hỏi CHỈ dựa trên context được cung cấp. "
                        "Nếu context không chứa đủ thông tin → nói 'Không tìm thấy thông tin trong tài liệu.' "
                        "Trả lời ngắn gọn, súc tích, chính xác. Trả lời bằng tiếng Việt."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"
                }
            ],
            max_tokens=200,
            temperature=0.1,  # Thấp → faithful hơn, ít hallucinate
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM generation failed: {e}")
        return contexts[0] if contexts else "Không tìm thấy thông tin."


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    # Search (M2: Hybrid BM25 + Dense)
    t0 = time.perf_counter()
    results = search.search(query)
    _latency["search"] += (time.perf_counter() - t0) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    # Rerank (M3: Cross-encoder)
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    _latency["rerank"] += (time.perf_counter() - t0) * 1000

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # Generate answer (LLM)
    t0 = time.perf_counter()
    answer = generate_answer(query, contexts)
    _latency["generate"] += (time.perf_counter() - t0) * 1000

    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Bonus: Latency breakdown report
    n = len(test_set)
    print("\n[Latency Breakdown]")
    print(f"  Chunking:   {_latency['chunk']:.0f}ms (one-time)")
    print(f"  Enrichment: {_latency['enrich']:.0f}ms (one-time)")
    print(f"  Indexing:   {_latency['index']:.0f}ms (one-time)")
    print(f"  Search:     {_latency['search']/n:.0f}ms avg/query")
    print(f"  Reranking:  {_latency['rerank']/n:.0f}ms avg/query")
    print(f"  Generation: {_latency['generate']/n:.0f}ms avg/query")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    # Lưu latency vào report
    import json
    latency_data = {
        "chunk_ms": round(_latency["chunk"], 1),
        "enrich_ms": round(_latency["enrich"], 1),
        "index_ms": round(_latency["index"], 1),
        "avg_search_ms": round(_latency["search"] / max(n, 1), 1),
        "avg_rerank_ms": round(_latency["rerank"] / max(n, 1), 1),
        "avg_generate_ms": round(_latency["generate"] / max(n, 1), 1),
    }
    with open("latency_report.json", "w") as f:
        json.dump(latency_data, f, indent=2)

    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
