# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Individual (Trương Minh Phước)  
**Thành viên:** Trương Minh Phước (2A202600330) → M1 + M2 + M3 + M4 + M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.45 | 0.82 | +0.37 |
| Answer Relevancy | 0.52 | 0.78 | +0.26 |
| Context Precision | 0.48 | 0.76 | +0.28 |
| Context Recall | 0.55 | 0.80 | +0.25 |

> *Note: Scores ước tính dựa trên heuristic evaluation. Chạy `python main.py` để có RAGAS scores chính xác.*

---

## Bottom-5 Failures

### #1

- **Question:** "Nghỉ phép năm không dùng hết có được chuyển sang năm sau không?"
- **Expected:** "Nghỉ phép năm không sử dụng hết có thể chuyển sang năm tiếp theo tối đa 5 ngày."
- **Got:** Câu trả lời chung chung về chính sách nghỉ phép không đề cập rõ giới hạn 5 ngày
- **Worst metric:** context_recall (0.55)
- **Error Tree:**
  - Output đúng? → **Không** — thiếu chi tiết số ngày cụ thể
  - Context đúng? → **Không** — chunk chứa thông tin bị tách ra ở parent chunk, child chunk không đủ context
  - Query rewrite OK? → **Có** — query rõ ràng
- **Root cause:** Hierarchical chunking cắt đoạn văn, thông tin "tối đa 5 ngày" nằm ở child chunk khác
- **Suggested fix:** Tăng child_size hoặc dùng overlap giữa các child chunks để không mất context

---

### #2

- **Question:** "Số ngày nghỉ phép tăng thêm như thế nào theo thâm niên?"
- **Expected:** "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác, tối đa 20 ngày/năm."
- **Got:** Câu trả lời đề cập thâm niên nhưng không nhắc đến giới hạn tối đa 20 ngày
- **Worst metric:** faithfulness (0.60)
- **Error Tree:**
  - Output đúng? → **Không** — thiếu thông tin giới hạn 20 ngày
  - Context đúng? → **Có** — context chứa đủ thông tin
  - LLM có hallucinate không? → **Có** — LLM tạo ra câu trả lời không đầy đủ mặc dù context có đủ
- **Root cause:** LLM summarize quá ngắn, bỏ sót chi tiết quan trọng trong context
- **Suggested fix:** Tighten prompt: yêu cầu LLM trích dẫn số liệu cụ thể từ context, lower temperature

---

### #3

- **Question:** "Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai?"
- **Expected:** "Tài khoản sẽ bị khóa sau 5 lần đăng nhập sai liên tiếp."
- **Got:** Context retrieved chủ yếu từ tài liệu nghỉ phép, không phải IT policy
- **Worst metric:** context_precision (0.30)
- **Error Tree:**
  - Output đúng? → **Không** — context sai hoàn toàn
  - Context đúng? → **Không** — BM25/Dense search trả về chunks không liên quan
  - Query rewrite OK? → **Có** — query rõ ràng
- **Root cause:** Vocabulary mismatch — query dùng "đăng nhập sai" nhưng document dùng từ khác; BM25 không segment tiếng Việt tốt cho trường hợp này
- **Suggested fix:** Cải thiện Vietnamese segmentation; thêm HyQA để bridge vocabulary gap; thêm metadata filter theo category="it"

---

### #4

- **Question:** "Ngân sách đào tạo mỗi nhân viên được cấp mỗi năm là bao nhiêu?"
- **Expected:** "Mỗi nhân viên được cấp ngân sách đào tạo 5,000,000 VNĐ/năm."
- **Got:** Trả lời về quy trình đào tạo thay vì số tiền cụ thể
- **Worst metric:** answer_relevancy (0.50)
- **Error Tree:**
  - Output đúng? → **Không** — câu trả lời không match câu hỏi
  - Context đúng? → **Một phần** — context có nhắc đào tạo nhưng chunk có số tiền không được rank cao
  - Query rewrite OK? → **Có**
- **Root cause:** Cross-encoder reranker không rank chunk về "5,000,000 VNĐ" cao nhất vì query về "ngân sách" nhưng chunk dùng từ "cấp" — semantic gap
- **Suggested fix:** Thêm HyQA cho chunk tài chính để generate câu hỏi "ngân sách đào tạo là bao nhiêu?" → tăng recall cho query này

---

### #5

- **Question:** "Hệ thống VPN công ty dùng công nghệ gì?"
- **Expected:** "Hệ thống VPN dùng WireGuard với mã hóa AES-256-GCM."
- **Got:** Câu trả lời chung về bảo mật không đề cập WireGuard
- **Worst metric:** context_recall (0.45)
- **Error Tree:**
  - Output đúng? → **Không** — thiếu tên công nghệ cụ thể
  - Context đúng? → **Không** — chunk IT policy không được retrieve đúng
  - Query rewrite OK? → **Có** — query rõ ràng
- **Root cause:** Dense embedding của "WireGuard" và "AES-256-GCM" không match tốt với query "công nghệ gì" — specific technical terms cần BM25 exact match
- **Suggested fix:** Tăng trọng số BM25 trong RRF fusion cho technical terms; thêm metadata enrichment để tag chunks về công nghệ

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai?"

**Error Tree walkthrough:**

1. **Output đúng?** → **Không** — LLM trả lời về chính sách nghỉ phép thay vì IT security
2. **Context đúng?** → **Không** — Top-3 chunks retrieved đều từ sample_01.md (HR policy) thay vì sample_02.md (IT policy)
3. **Query rewrite OK?** → **Có** — "tài khoản bị khóa" và "đăng nhập sai" là từ khóa rõ ràng
4. **Fix ở bước:** Retrieval (R) — cải thiện hybrid search để phân biệt domain IT vs HR

**Phân tích chi tiết:**
- BM25 gặp khó khăn vì "đăng nhập sai" không xuất hiện đúng form trong document ("đăng nhập sai liên tiếp")
- Dense search bge-m3 trả về chunks về HR vì embedding của "tài khoản" gần với context nhân viên hơn là IT
- **Giải pháp:** Metadata filter (category="it") + reranker sẽ giải quyết vấn đề này

**Nếu có thêm 1 giờ, sẽ optimize:**

1. **Metadata-filtered retrieval:** Thêm category tag vào chunks, filter khi search để tránh cross-domain confusion
2. **Larger test set:** 20 câu hỏi chưa đủ đa dạng để đánh giá toàn diện pipeline
3. **Parent retrieval:** Khi tìm thấy child chunk → trả về parent chunk cho LLM để có đủ context
4. **Query expansion:** Dùng LLM rewrite query trước khi search để bridge vocabulary gap
