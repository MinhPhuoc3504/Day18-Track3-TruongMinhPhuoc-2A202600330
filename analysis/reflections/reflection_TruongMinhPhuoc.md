# Individual Reflection — Lab 18

**Tên:** Trương Minh Phước  
**MSSV:** 2A202600330  
**Module phụ trách:** M1 + M2 + M3 + M4 + M5 (tất cả modules)

---

## 1. Đóng góp kỹ thuật

**Module đã implement:**
- **M1 (Advanced Chunking):** chunk_semantic(), chunk_hierarchical(), chunk_structure_aware(), compare_strategies()
- **M2 (Hybrid Search):** segment_vietnamese(), BM25Search.index/search(), DenseSearch.index/search(), reciprocal_rank_fusion()
- **M3 (Reranking):** CrossEncoderReranker._load_model/rerank(), FlashrankReranker.rerank(), benchmark_reranker()
- **M4 (RAGAS Evaluation):** evaluate_ragas(), failure_analysis(), _heuristic_evaluate() (fallback)
- **M5 (Enrichment Pipeline):** summarize_chunk(), generate_hypothesis_questions(), contextual_prepend(), extract_metadata(), enrich_chunks()
- **Pipeline:** build_pipeline(), run_query(), evaluate_pipeline(), generate_answer() với LLM

**Các hàm/class chính đã viết:**
- 20+ functions/methods across 5 modules
- Data files: sample_01.md, sample_02.md, sample_03.md (3 domains: HR, IT, Recruitment)
- Test set: 20 câu hỏi Q&A thực tế về chính sách công ty

**Số tests pass:** 29/29 (100%) — tất cả tests từ test_m1.py đến test_m5.py

---

## 2. Kiến thức học được

**Khái niệm mới nhất:**

**Reciprocal Rank Fusion (RRF):** Cách merge nhiều ranked lists thành 1 list tốt hơn. Công thức score(d) = Σ 1/(k+rank_i) đơn giản nhưng hiệu quả — document xuất hiện cao trong nhiều lists → score cao. k=60 là magic number từ paper gốc (Cormack et al. 2009).

**Contextual Prepend (Anthropic):** Thêm 1 câu context "chunk này nằm ở đâu trong document" vào trước mỗi chunk trước khi embed. Nghe đơn giản nhưng theo Anthropic benchmark giảm 49% retrieval failure. Bản chất: embedding model nhìn full context → vector chính xác hơn.

**Hierarchical Chunking:** Parent (2048 chars) + Child (256 chars). Index children (nhỏ → embedding chính xác) nhưng trả parent về cho LLM (đủ context). Pattern production-ready nhất cho RAG.

**Điều bất ngờ nhất:**

Vietnamese segmentation quan trọng hơn tôi nghĩ. "nghỉ_phép" là 1 token trong BM25, còn "nghỉ" và "phép" riêng lẻ cho kết quả search hoàn toàn khác. underthesea giải quyết vấn đề này rất tốt cho tiếng Việt.

**Kết nối với bài giảng:**

- Slide về "Why RAG fails" → trực tiếp áp dụng vào failure_analysis() với Diagnostic Tree
- Slide về "Chunking strategies" → implement đủ 3 loại (semantic, hierarchical, structure-aware)
- Slide về "Hybrid Search" → BM25 + Dense + RRF là pipeline chuẩn production

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn 1: RAGAS evaluation timeout**

RAGAS thực gọi OpenAI API cho mỗi metric của mỗi câu hỏi → với 20 câu hỏi × 4 metrics = 80 API calls → chậm và tốn tiền.

Giải pháp: Implement `_heuristic_evaluate()` làm fallback. Dùng word overlap để tính approximate scores → nhanh hơn nhiều, không cần API, vẫn cho insights về pipeline performance.

**Khó khăn 2: Qdrant connection**

DenseSearch cần Qdrant server running. Nếu không có Docker, pipeline bị fail ngay từ bước index.

Giải pháp: Wrap trong try/except, trả về empty list nếu Qdrant không available. BM25 vẫn hoạt động standalone → pipeline gracefully degraded.

**Khó khăn 3: bge-reranker-v2-m3 download chậm**

Model 1.1GB → first load mất 5-10 phút, test sẽ fail do timeout.

Giải pháp: Fallback về `cross-encoder/ms-marco-MiniLM-L-6-v2` (nhỏ hơn) khi bge không load được. Vẫn đảm bảo reranking hoạt động.

**Thời gian debug:** ~2 giờ tổng (chủ yếu cho RAGAS integration và Qdrant setup)

---

## 4. Nếu làm lại

**Sẽ làm khác:**

1. **Parent retrieval:** Implement lookup parent từ child_id ngay từ đầu. Hiện tại pipeline trả child text cho LLM → thiếu context. Cần: khi retrieve child → trả về parent text để LLM có đủ thông tin.

2. **Chunk overlap:** Thêm 10-20% overlap giữa các child chunks để không mất context ở ranh giới chunk. Hiện tại hard split có thể cắt đứt câu.

3. **Metadata-based filtering:** Thêm category filter (hr/it/finance) vào search. Tránh trường hợp query về IT security retrieve được HR policy chunks.

4. **Async enrichment:** Enrichment dùng OpenAI API → chạy tuần tự chậm. Nên dùng `asyncio` để gọi API song song → nhanh hơn 5-10x.

**Module muốn thử tiếp:**

**Query expansion/rewriting:** Trước khi search, dùng LLM viết lại query thành nhiều dạng khác nhau → search với tất cả variants → merge results. Đặc biệt hữu ích cho câu hỏi ngắn hoặc ambiguous.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Lý do |
|----------|---------------|-------|
| Hiểu bài giảng | 5 | Implement được tất cả concepts từ bài giảng vào code |
| Code quality | 4 | Clean, typed, commented. Trừ 1 vì một số edge cases chưa handle |
| Teamwork | 4 | Làm individual nhưng code được viết để dễ integrate với team |
| Problem solving | 5 | Giải quyết được tất cả khó khăn kỹ thuật, có fallback cho mọi bước |

**Tổng tự đánh giá: 4.5/5**

Bài lab này giúp tôi hiểu tại sao RAG production khác hoàn toàn với RAG demo. Mỗi module (chunking, search, reranking, evaluation) đều có nhiều lựa chọn với trade-off khác nhau. Failure analysis là phần thú vị nhất vì buộc phải suy nghĩ có hệ thống về "tại sao" thay vì chỉ "cái gì".
