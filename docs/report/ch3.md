# CHƯƠNG 3. CÔNG NGHỆ SỬ DỤNG

Chương này trình bày một cách có hệ thống các công nghệ được lựa chọn để xây dựng hệ thống VITAL (AgentRag) — chatbot hỏi đáp tài liệu y tế tiếng Việt dựa trên kiến trúc RAG (Retrieval-Augmented Generation — sinh câu trả lời có tăng cường truy xuất) kết hợp tác tử (agent). Với mỗi công nghệ, nội dung được trình bày theo trình tự: công nghệ đó là gì, nó giải quyết bài toán nào trong hệ thống, và vì sao nó được chọn thay cho các phương án cạnh tranh. Các phiên bản, tên mô hình và tham số cấu hình nêu trong chương đều là giá trị thực tế được khai báo trong mã nguồn của dự án (tệp `pyproject.toml`, `docker-compose.yml`, `.env.example`, `src/agentrag/config.py`) tại thời điểm viết báo cáo.

## 3.1. Ngôn ngữ và framework backend

### 3.1.1. Python 3.11+

Toàn bộ backend của hệ thống được viết bằng Python, với yêu cầu phiên bản tối thiểu được khai báo tường minh trong `pyproject.toml` là `requires-python = ">=3.11"`. Python được chọn làm ngôn ngữ chủ đạo vì ba lý do.

Thứ nhất, hệ sinh thái học máy và xử lý ngôn ngữ tự nhiên của Python là phong phú nhất hiện nay: các thư viện then chốt của đề tài như `sentence-transformers` (mã hóa và huấn luyện mô hình embedding), `langgraph` (điều phối tác tử), `faster-whisper` (nhận dạng tiếng nói), `pymupdf` (phân tích PDF) đều là thư viện Python hoặc có giao diện Python hạng nhất. Chọn một ngôn ngữ khác đồng nghĩa với việc phải tự cài cầu nối cho từng thư viện này.

Thứ hai, từ phiên bản 3.11, bộ thông dịch CPython có bước nhảy đáng kể về hiệu năng (dự án Faster CPython của Python Software Foundation công bố tốc độ trung bình nhanh hơn 10–60% so với 3.10) và về trải nghiệm phát triển: thông báo lỗi chỉ đích danh biểu thức gây lỗi, nhóm ngoại lệ (`ExceptionGroup`) phục vụ lập trình bất đồng bộ, và cú pháp gợi ý kiểu (type hint) hoàn thiện hơn. Hệ thống VITAL là ứng dụng thiên về vào/ra (I/O-bound): phần lớn thời gian xử lý một lượt hỏi đáp là chờ Elasticsearch, PostgreSQL và các API mô hình ngôn ngữ trả lời. Mô hình lập trình bất đồng bộ `asyncio` của Python cho phép một tiến trình phục vụ nhiều yêu cầu đồng thời mà không cần nhiều luồng hệ điều hành, phù hợp tự nhiên với đặc điểm đó.

Thứ ba, dự án dùng `uv` — trình quản lý gói và môi trường ảo thế hệ mới viết bằng Rust — để cài đặt phụ thuộc (`uv sync --frozen` trong Dockerfile, khóa phiên bản bằng `uv.lock`). Nhờ đó việc dựng lại môi trường trên máy phát triển, trên CI và trong ảnh Docker là tái lập được (reproducible): cùng một tệp khóa cho ra cùng một tập phiên bản thư viện.

### 3.1.2. FastAPI — framework web bất đồng bộ

FastAPI (dự án khai báo `fastapi[standard]>=0.115.0`) là framework web hiện đại của Python xây trên chuẩn ASGI (Asynchronous Server Gateway Interface — giao diện cổng máy chủ bất đồng bộ), thay cho chuẩn WSGI đồng bộ truyền thống. FastAPI đảm nhận toàn bộ tầng adapter của hệ thống: các router `/chat`, `/search`, `/sources`, `/metrics/cost`… đều là ứng dụng FastAPI, chạy sau máy chủ Gunicorn với 4 worker lớp `uvicorn.workers.UvicornWorker` (khai báo trong `Dockerfile`: `gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 main:app --timeout 120`).

Lý do chọn FastAPI thay vì các framework Python phổ biến khác được tóm tắt trong bảng so sánh sau:

| Tiêu chí | FastAPI | Flask | Django |
|---|---|---|---|
| Mô hình thực thi | Bất đồng bộ gốc (ASGI, `async def`) | Đồng bộ (WSGI); async là phần vá thêm | Đồng bộ là chính; ASGI hỗ trợ một phần |
| Kiểm tra dữ liệu vào/ra | Tự động qua Pydantic, khai báo bằng type hint | Thủ công hoặc qua extension (Marshmallow…) | Django Forms/Serializers, tách rời type hint |
| Tài liệu API | Sinh tự động OpenAPI/Swagger tại `/docs` | Cần extension | Cần Django REST framework + công cụ phụ |
| Streaming SSE | Hỗ trợ trực tiếp qua `StreamingResponse` | Khó, worker đồng bộ bị chiếm giữ suốt phiên | Tương tự Flask khi chạy WSGI |
| Độ phù hợp với ứng dụng I/O-bound gọi LLM | Cao — hàng nghìn kết nối chờ trên ít tiến trình | Thấp — mỗi request chiếm một worker | Trung bình |

Hai tính năng của FastAPI mang tính quyết định với đề tài:

**Một là truyền phát SSE (Server-Sent Events — sự kiện do máy chủ đẩy xuống).** Khi mô hình ngôn ngữ sinh câu trả lời, người dùng cần thấy văn bản xuất hiện dần thay vì chờ 10–30 giây rồi nhận cả khối. Router chat của hệ thống trả về `StreamingResponse` với `media_type="text/event-stream"` (mã nguồn `src/agentrag/adapter/routers/chat.py`), đẩy từng phần câu trả lời xuống trình duyệt qua một kết nối HTTP duy nhất. Với framework đồng bộ như Flask, một phiên streaming 30 giây sẽ chiếm trọn một worker; với FastAPI/ASGI, worker chỉ bị chiếm tại các thời điểm thực sự có dữ liệu để ghi, phần thời gian chờ được nhường cho yêu cầu khác.

**Hai là kiểm tra kiểu dữ liệu tự động bằng Pydantic.** Mỗi endpoint khai báo schema vào/ra bằng lớp Pydantic; dữ liệu sai kiểu bị chặn ngay ở biên với thông báo lỗi 422 có cấu trúc, đồng thời tài liệu tương tác Swagger UI được sinh tự động tại `http://localhost:8000/docs` — đây cũng chính là giao diện mà nhóm dùng để kiểm thử thủ công API trong quá trình phát triển.

### 3.1.3. Pydantic Settings — quản lý cấu hình qua biến môi trường

Hệ thống có trên một trăm tham số cấu hình: địa chỉ các dịch vụ hạ tầng, tên mô hình cho từng tác vụ, ngưỡng an toàn của bộ truy xuất, cờ bật/tắt tính năng… Toàn bộ được quản lý tập trung bằng thư viện `pydantic-settings` (>=2.2.0) kết hợp `python-dotenv`: lớp `Settings` trong `src/agentrag/config.py` khai báo từng tham số kèm kiểu dữ liệu và giá trị mặc định, còn giá trị thực tế được nạp từ tệp `.env` hoặc biến môi trường của tiến trình.

Cách tiếp cận này tuân theo nguyên tắc "cấu hình tách khỏi mã nguồn" của phương pháp luận Twelve-Factor App: cùng một ảnh Docker chạy được ở máy phát triển, CI và máy chủ chỉ bằng cách thay tệp `.env`. Ưu thế của Pydantic Settings so với đọc `os.environ` thủ công là **kiểm tra kiểu tại thời điểm khởi động**: một giá trị sai kiểu hoặc sai miền (ví dụ gán chuỗi tùy ý vào trường khai báo `Literal["llm_chat", "local_cross_encoder"]`) làm ứng dụng từ chối khởi động ngay với thông báo rõ ràng, thay vì gây lỗi ngầm lúc chạy. Một trích đoạn cấu hình thực tế của hệ thống:

```bash
# .env — cấu hình vận hành thực tế (docs/HOME-RUN.md)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=agentrag-embed-v1
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/
EMBEDDING_OUTPUT_DIM=768
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=BAAI/bge-reranker-v2-m3
OBSERVABILITY_CAPTURE_CONTENT=false
```

### 3.1.4. SQLAlchemy async và Alembic — tầng truy cập dữ liệu quan hệ

Tầng truy cập PostgreSQL dùng SQLAlchemy phiên bản 2.x ở chế độ bất đồng bộ (`sqlalchemy[asyncio]>=2.0.0`) với driver `psycopg` thế hệ 3 (`psycopg[binary]>=3.1.0`). SQLAlchemy là ORM (Object-Relational Mapper — bộ ánh xạ đối tượng-quan hệ) tiêu chuẩn của Python: các bảng `documents`, `segments`, `conversations`, `messages`, `event_log`… được khai báo thành lớp Python trong `database/models.py`, và mọi truy vấn đi qua phiên `AsyncSessionLocal` để không chặn vòng lặp sự kiện của FastAPI — nhất quán với lựa chọn kiến trúc bất đồng bộ xuyên suốt ở mục 3.1.2.

Đi kèm SQLAlchemy là **Alembic** (>=1.13.0) — công cụ di trú lược đồ (schema migration). Mỗi thay đổi cấu trúc bảng được ghi thành một tệp migration có thứ tự, giúp cơ sở dữ liệu ở mọi môi trường tiến hóa cùng nhịp với mã nguồn bằng một lệnh duy nhất (`make migrate`, tức `alembic upgrade head`). Đây là yêu cầu bắt buộc với một hệ thống có dữ liệu người dùng thật: không thể xóa-dựng lại cơ sở dữ liệu mỗi lần đổi lược đồ như giai đoạn thử nghiệm.

## 3.2. Lưu trữ dữ liệu

Hệ thống dùng ba hệ lưu trữ chuyên biệt hóa, mỗi hệ giữ đúng vai trò mà nó mạnh nhất: PostgreSQL là nguồn sự thật (source of truth), Elasticsearch là máy tìm kiếm, và Valkey (Redis) là bộ đệm kiêm hàng đợi tác vụ. Cả ba được khai báo trong `docker-compose.yml` và khởi động bằng một lệnh `make docker-up`.

### 3.2.1. PostgreSQL — nguồn sự thật

PostgreSQL (ảnh `postgres:16-alpine`, ánh xạ cổng máy chủ 5433 → 5432 trong container) là hệ quản trị cơ sở dữ liệu quan hệ mã nguồn mở, lưu **bản chính** của mọi dữ liệu có cấu trúc: tài liệu gốc và siêu dữ liệu của chúng (`documents`), các đoạn văn bản đã cắt (`segments`), hội thoại và tin nhắn chat, phản hồi 👍/👎 của người dùng, nhật ký sự kiện `event_log`, tài khoản người dùng. Elasticsearch có thể xây lại từ PostgreSQL bất cứ lúc nào (re-index), nhưng chiều ngược lại thì không — do đó chỉ PostgreSQL cần được sao lưu nghiêm ngặt.

PostgreSQL được chọn thay vì MySQL hay các cơ sở dữ liệu NoSQL vì: (1) tính giao dịch ACID đầy đủ, cần thiết khi một thao tác xóa tài khoản phải xóa nguyên tử toàn bộ dữ liệu liên quan (chức năng right-to-delete `DELETE /chat/account` của hệ thống); (2) kiểu dữ liệu phong phú (JSONB, UUID, mảng) khớp với mô hình dữ liệu của ứng dụng; (3) hệ sinh thái phần mở rộng — môi trường CI của dự án chạy ảnh `pgvector/pgvector:pg16` vì lược đồ có dùng phần mở rộng pgvector (lưu vector trong cơ sở dữ liệu quan hệ); (4) driver bất đồng bộ trưởng thành (`psycopg` 3) khớp với ngăn xếp asyncio.

### 3.2.2. Elasticsearch — tìm kiếm lai BM25 + kNN với RRF

Elasticsearch (client và server đều khóa phiên bản `8.15.0`, chạy chế độ một nút `discovery.type=single-node`, heap 1 GB) là máy tìm kiếm phân tán xây trên thư viện Apache Lucene. Trong hệ thống, nó đảm nhận toàn bộ khâu truy xuất của RAG với các chỉ mục (index) thực tế:

- `agentrag_segments` — chỉ mục chính chứa các đoạn tài liệu y tế đã cắt, mỗi đoạn gồm trường văn bản (phục vụ BM25) và trường vector dày đặc 768 chiều (phục vụ kNN);
- `agentrag_memory_doc` — chỉ mục bộ nhớ có cấu trúc (StructMem) của tác tử, cấu hình qua `STRUCTMEM_INDEX`;
- `agentrag_memory_chat` — chỉ mục bộ nhớ hội thoại (nêu trong chú thích `.env.example`).

Ba kỹ thuật truy xuất được phối hợp:

**BM25** (Best Matching 25) là hàm xếp hạng từ vựng cổ điển: điểm của một văn bản tỉ lệ với tần suất từ khóa truy vấn xuất hiện trong nó (TF), được chuẩn hóa theo độ hiếm của từ trong toàn kho (IDF) và theo độ dài văn bản. BM25 mạnh với các truy vấn chứa thuật ngữ chính xác — tên thuốc, mã bệnh, liều lượng — vốn rất phổ biến trong miền y tế.

**Tìm kiếm kNN dày đặc** (dense k-nearest-neighbor) so khớp vector embedding của câu hỏi với vector của từng đoạn bằng độ tương đồng cosine, dựa trên cấu trúc chỉ mục HNSW của Lucene. Cách này bắt được quan hệ ngữ nghĩa mà BM25 bỏ lỡ: "huyết áp cao" và "tăng huyết áp" là hai chuỗi ký tự khác nhau nhưng có vector gần nhau.

**RRF** (Reciprocal Rank Fusion — hợp nhất theo nghịch đảo thứ hạng) trộn hai danh sách kết quả trên: mỗi văn bản nhận điểm `Σ 1/(k + hạng_i)` cộng trên các danh sách nó xuất hiện, với hằng số `k = 60` (tham số `RETRIEVAL_RRF_K` trong `config.py`). RRF không cần chuẩn hóa thang điểm giữa BM25 và cosine — vốn không so sánh được trực tiếp — mà chỉ dùng thứ hạng, nên bền vững và không cần tinh chỉnh.

So sánh với các phương án lưu trữ vector chuyên dụng:

| Tiêu chí | Elasticsearch 8.x | FAISS | Qdrant |
|---|---|---|---|
| Bản chất | Máy tìm kiếm đầy đủ (Lucene) | Thư viện tìm vector trong tiến trình | Cơ sở dữ liệu vector chuyên dụng |
| BM25 từ vựng | Có, hạng nhất | Không | Không (chỉ lọc payload; full-text hạn chế) |
| Lai BM25 + vector trong một hệ | Có, một truy vấn | Phải tự ghép với hệ khác | Phải tự ghép BM25 ngoài |
| Lọc theo siêu dữ liệu (hệ cơ quan, chuyên khoa, quyền truy cập) | Truy vấn bool tùy ý | Rất hạn chế | Tốt |
| Bền vững dữ liệu, quản trị | Đầy đủ (snapshot, replica) | Không (tự lo tuần tự hóa) | Đầy đủ |
| Thêm một dịch vụ hạ tầng mới | Không cần nếu đã dùng ES | Không (nhúng) | Cần |

FAISS chỉ là thư viện: nó tìm vector rất nhanh nhưng không có tầng bền vững, không có bộ lọc siêu dữ liệu, và hoàn toàn không có BM25 — trong khi truy xuất y tế cần cả khớp thuật ngữ chính xác lẫn lọc theo hệ cơ quan/chuyên khoa (bộ lọc `systems`/`specialties` của `RetrievalService`). Qdrant làm tốt phần vector nhưng vẫn phải ghép thêm một máy tìm kiếm từ vựng riêng cho BM25, làm kiến trúc phình thành hai hệ. Elasticsearch cho cả ba năng lực (BM25, kNN, lọc bool) trong **một** dịch vụ, một truy vấn — đó là lý do quyết định.

### 3.2.3. Valkey (Redis) — bộ đệm và hàng đợi tác vụ ARQ

Valkey (ảnh `valkey/valkey:8-alpine`) là nhánh mã nguồn mở của Redis, ra đời năm 2024 dưới sự bảo trợ của Linux Foundation sau khi Redis Inc. đổi sang giấy phép hạn chế; Valkey giữ giấy phép BSD và tương thích hoàn toàn giao thức Redis, nên phía Python vẫn dùng client `redis>=5.0.0` bình thường (`REDIS_URL=redis://valkey:6379/0`). Đây là kho khóa-giá trị trong bộ nhớ, được hệ thống dùng cho ba việc:

1. **Hàng đợi tác vụ ARQ.** ARQ (`arq>=0.26.0`) là thư viện hàng đợi tác vụ bất đồng bộ gọn nhẹ xây trên Redis. Các công việc nặng và chậm — phân tích tài liệu tải lên, OCR, sinh embedding hàng loạt, đánh chỉ mục — không được xử lý trong tiến trình API (sẽ làm nghẽn yêu cầu chat) mà được đẩy vào hàng đợi để container `worker` riêng xử lý (lệnh khởi động thực tế trong compose: `arq src.agentrag.worker.settings.WorkerSettings`). ARQ được chọn thay Celery vì nó là async-native — worker dùng lại nguyên vẹn các service bất đồng bộ của ứng dụng — và cấu hình tối giản, trong khi Celery đồng bộ gốc và kéo theo nhiều tầng trừu tượng không cần thiết ở quy mô đề tài.
2. **Sổ cái chi phí LLM.** Mỗi lời gọi mô hình ngôn ngữ thành công được ghi một bản ghi vào Redis Stream `agentrag:llm:calls:v1` (giới hạn `MAXLEN ~ 5000`), sống sót qua khởi động lại và gộp được số liệu giữa nhiều worker (chi tiết ở mục 3.8).
3. **Bộ đệm.** Các kết quả trung gian được đệm để giảm gọi lặp (kết hợp với các cache trong tiến trình như cache embedding TTL và semantic cache nêu ở mục 3.4).

Valkey chạy với `--appendonly yes` (ghi nhật ký AOF) để dữ liệu hàng đợi và sổ cái không mất khi container khởi động lại.

## 3.3. Mô hình ngôn ngữ lớn

### 3.3.1. Bài toán: không một mô hình nào tối ưu cho mọi tác vụ

Một lượt hỏi đáp của tác tử không phải là một lời gọi LLM duy nhất mà là một chuỗi tác vụ khác nhau về bản chất: phân loại câu hỏi có phải chitchat không, quyết định bước tiếp theo (decide), lập kế hoạch truy xuất (plan), tổng hợp câu trả lời có trích dẫn (answer), sinh câu hỏi gợi ý tiếp theo (followup), chấm điểm chất lượng khi đánh giá (eval_judge)… Các tác vụ này có yêu cầu rất khác nhau: tác vụ định tuyến được gọi nhiều lần mỗi lượt nhưng chỉ cần đầu ra JSON ngắn đúng cấu trúc; tác vụ trả lời cần năng lực suy luận và tổng hợp mạnh; tác vụ chấm điểm cần độc lập với mô hình sinh câu trả lời để tránh thiên vị. Dùng một mô hình lớn duy nhất cho tất cả vừa đắt vừa chậm; dùng một mô hình nhỏ duy nhất thì chất lượng câu trả lời không đạt.

Hệ thống giải quyết bằng **LLM Gateway** (`src/agentrag/services/llm_gateway.py`) — điểm gọi LLM duy nhất của toàn bộ mã nguồn. Mọi nơi cần LLM (tác tử, sinh câu trả lời, ingestion) đều gọi qua gateway với một tham số `task`; gateway chịu trách nhiệm chọn mô hình, chọn nhà cung cấp, đo độ trễ và ghi chi phí. Tất cả nhà cung cấp được truy cập qua giao thức OpenAI-compatible (SDK `openai>=1.30.0` với `base_url` khác nhau), nên việc thêm/đổi nhà cung cấp không đụng đến mã nghiệp vụ.

### 3.3.2. Các nhà cung cấp và mô hình được sử dụng

**Ollama (tự vận hành, chạy local).** Ollama là runtime mã nguồn mở chạy LLM trên máy cá nhân, phơi API tương thích OpenAI tại cổng 11434. Hệ thống dùng **`llama3.2:3b`** — mô hình 3 tỷ tham số của Meta — cho toàn bộ các tác vụ điều phối nhẹ (`classify`, `decide`, `domain_router`, `followup`). Đây là các lời gọi tần suất cao, yêu cầu thấp: chạy local nghĩa là chi phí bằng không, độ trễ ổn định, và nội dung câu hỏi không rời khỏi máy. Cấu hình compose đặt `OLLAMA_KEEP_ALIVE=24h` (giữ mô hình trong VRAM 24 giờ sau lần dùng cuối, tránh chi phí nạp lại) và `OLLAMA_MAX_LOADED_MODELS=3` (GPU 16 GB đủ giữ đồng thời `qwen2.5:7b`, `llava:7b` và `llama3.2:3b`). Mô hình `qwen2.5:7b-instruct` được khai là `LLM_FALLBACK_MODEL` — phương án dự phòng khi mô hình chính không khả dụng.

**DeepSeek (API thương mại, giao thức OpenAI-compatible, `base_url=https://api.deepseek.com`).** Các tác vụ nặng được đẩy lên hai mô hình: **`deepseek-v4-pro`** cho các tác vụ cần suy luận sâu (`plan`, `schema_discovery`, `sql_compile`) và **`deepseek-v4-flash`** — bản nhanh, rẻ hơn — cho các tác vụ sinh nội dung khối lượng lớn (`answer`, `synthesize`, `mindmap`, `summary`). DeepSeek được chọn làm "ngựa thồ" chính vì tỉ lệ chất lượng/chi phí tốt trên các tác vụ sinh văn bản dài, và vì giao thức OpenAI-compatible cho phép hoán đổi không ma sát.

**Google Gemini 2.5.** Hai mô hình `gemini-2.5-pro` và `gemini-2.5-flash` giữ hai vai trò: (1) phương án đám mây thay thế hoàn chỉnh (tệp `.env.example` cung cấp sẵn một biến thể `LLM_TASK_MODEL_MAP` toàn Gemini); (2) quan trọng hơn, làm **giám khảo đánh giá độc lập** (`eval_judge`) trong hạ tầng đo lường chất lượng: vì mô hình trả lời là DeepSeek, giám khảo phải thuộc nhà cung cấp khác để loại thiên vị tự ưu ái (self-preference). Gemini 2.5 Pro còn có cửa sổ ngữ cảnh 1 triệu token, phù hợp làm mô hình định tuyến ngữ cảnh lớn (`LLM_LARGE_CONTEXT_MODEL`).

**Anthropic Claude.** Provider `anthropic` được nối dây sẵn trong `agent/llm.py` (mô hình tên `claude-*` tự định tuyến về `https://api.anthropic.com/v1/` khi có `ANTHROPIC_API_KEY`); cấu hình mẫu dùng `claude-haiku-4-5` làm `eval_judge` — một giám khảo độc lập thay thế cho Gemini trả phí.

### 3.3.3. Cơ chế LLM_TASK_MODEL_MAP — định tuyến mô hình theo tác vụ

Trái tim của gateway là tham số `LLM_TASK_MODEL_MAP` — một ánh xạ JSON từ tên tác vụ sang tên mô hình, bật bằng cờ `LLM_ROUTING_ENABLED`. Cấu hình mặc định thực tế trong `.env.example`:

```bash
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b",
  "domain_router":"llama3.2:3b","followup":"llama3.2:3b",
  "plan":"deepseek-v4-pro","schema_discovery":"deepseek-v4-pro",
  "sql_compile":"deepseek-v4-pro","synthesize":"deepseek-v4-flash",
  "answer":"deepseek-v4-flash","mindmap":"deepseek-v4-flash",
  "summary":"deepseek-v4-flash"}
```

Cấu hình phục vụ đánh giá bổ sung thêm các khe tác vụ `oracle_gen`, `gold_gen`, `eval_judge`, `eval_judge2` (ví dụ `eval_judge=gemini-2.5-pro`, `eval_judge2=deepseek-v4-pro` để đo mức đồng thuận giữa hai giám khảo khác nhà cung cấp).

Việc tách theo tác vụ mang lại bốn lợi ích đã kiểm chứng trong vận hành:

1. **Chi phí.** Các tác vụ định tuyến chiếm đa số lời gọi nhưng chạy trên `llama3.2:3b` local miễn phí; chỉ phần sinh câu trả lời mới tốn phí API.
2. **Độ trễ.** Mô hình 3B local trả lời các quyết định điều phối trong thời gian ngắn và không phụ thuộc mạng.
3. **Chất lượng đúng chỗ.** Tác vụ `plan` phức tạp nhận `deepseek-v4-pro`; tác vụ `answer` khối lượng lớn nhận bản `flash` — mỗi tác vụ được cấp đúng "cỡ" mô hình.
4. **Độc lập đánh giá.** Giám khảo chấm điểm thuộc nhà cung cấp khác mô hình trả lời — điều kiện phương pháp luận để điểm số đáng tin.

Ngoài định tuyến theo tác vụ, gateway còn hai tầng quyết định: **định tuyến ngữ cảnh lớn tự động** — khi ước lượng số token của prompt vượt `LLM_LARGE_CONTEXT_THRESHOLD` (mặc định 100.000), lời gọi được tự chuyển sang `LLM_LARGE_CONTEXT_MODEL` (ví dụ `gemini-2.5-pro`); và **cơ chế phòng hộ thời gian**: `LLM_REQUEST_TIMEOUT_S=60` giới hạn từng lời gọi, `AGENT_TOTAL_TIMEOUT_S=90` chặn trần toàn vòng lặp tác tử để một nhà cung cấp chậm không treo cả lượt chat.

## 3.4. Mô hình embedding

### 3.4.1. Embedding văn bản và kiến trúc bi-encoder

Mô hình embedding biến một đoạn văn bản thành một vector số thực có số chiều cố định sao cho các văn bản gần nghĩa có vector gần nhau (đo bằng độ tương đồng cosine). Đây là nền tảng của nhánh truy xuất dày đặc: lúc nạp tài liệu, mỗi đoạn được mã hóa một lần và lưu vector vào Elasticsearch; lúc truy vấn, câu hỏi được mã hóa và so với toàn bộ kho bằng kNN. Kiến trúc này gọi là **bi-encoder** (bộ mã hóa kép): câu hỏi và văn bản được mã hóa *độc lập*, nhờ đó phần tốn kém (mã hóa kho tài liệu) làm trước được một lần — đổi lại độ chính xác thấp hơn cross-encoder, điểm sẽ bàn ở mục 3.5.

Với tiếng Việt y tế, truy xuất dày đặc đặc biệt quan trọng vì cùng một khái niệm có nhiều cách diễn đạt (thuật ngữ Hán-Việt, thuần Việt, tên tiếng Anh, viết tắt) mà BM25 thuần từ vựng không bắt được.

### 3.4.2. Các mô hình nền: multilingual-e5 và bge-m3

Hệ thống dùng hai họ mô hình embedding đa ngôn ngữ mã nguồn mở đã có kiểm chứng tốt trên tiếng Việt:

- **`intfloat/multilingual-e5-base`** — họ E5 của Microsoft, huấn luyện tương phản trên cặp văn bản quy mô lớn, hỗ trợ khoảng 100 ngôn ngữ, đầu ra 768 chiều. Điểm đặc thù của E5 là **quy ước tiền tố**: câu truy vấn phải mã hóa dưới dạng `query: ...`, đoạn tài liệu dưới dạng `passage: ...` — mô hình được huấn luyện phân biệt hai vai trò này, dùng sai tiền tố làm giảm chất lượng truy xuất rõ rệt.
- **`BAAI/bge-m3`** — mô hình của Beijing Academy of AI, nổi bật ở tính "3 trong 1" (dense, sparse, multi-vector) và cửa sổ ngữ cảnh dài. Trong hệ thống, bge-m3 là mô hình phục vụ mặc định của cấu hình compose gốc (TEI với `--pooling=cls`) và là phương án dự phòng khi triển khai mới.

### 3.4.3. agentrag-embed-v1 — mô hình embedding tinh chỉnh cho y tế tiếng Việt

Mô hình embedding đang phục vụ chính thức là **`agentrag-embed-v1`** — bản tinh chỉnh (fine-tune) của `intfloat/multilingual-e5-base` trên chính miền dữ liệu của đề tài. Theo model card của dự án (`deploy/model-card-agentrag-embed-v1.md`), các thông số của mô hình:

| Thuộc tính | Giá trị |
|---|---|
| Mô hình nền | `intfloat/multilingual-e5-base` |
| Số chiều đầu ra | 768 |
| Pooling | mean (trung bình các token) + chuẩn hóa L2 |
| Độ dài chuỗi tối đa | 512 token |
| Dữ liệu huấn luyện | 5.300 bộ ba (truy vấn, đoạn đúng, đoạn nhiễu) tiếng Việt y tế |
| Hàm mất mát | MultipleNegativesRankingLoss, 2 epoch |
| Kết quả | recall@10 tăng +0,20 so với mô hình nền trên benchmark truy xuất của dự án |

Mức cải thiện recall@10 +0,20 (tức trong 10 kết quả đầu, tỉ lệ tìm thấy đoạn chứa đáp án tăng thêm 20 điểm phần trăm) là "cổng thăng cấp" (promotion gate) của quy trình: mô hình tinh chỉnh chỉ được đưa vào phục vụ khi vượt mô hình nền trên tập kiểm thử tách riêng. Quy trình huấn luyện chi tiết trình bày ở mục 3.10.

### 3.4.4. Text Embeddings Inference — máy chủ phục vụ embedding

**TEI (Text Embeddings Inference)** là máy chủ suy luận embedding mã nguồn mở của Hugging Face, viết bằng Rust, chuyên phục vụ các mô hình embedding/reranking với hiệu năng cao: gom lô động (dynamic batching), quản lý VRAM chặt, và phơi API tương thích OpenAI (`/v1/embeddings`). Thay vì nạp mô hình vào tiến trình Python của API (chiếm VRAM của mọi worker, khởi động chậm), hệ thống chạy TEI như một container riêng tại cổng 8080; ứng dụng chỉ cần trỏ `EMBEDDING_PROVIDER=openai` và `EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/` — đúng cùng giao thức với mọi nhà cung cấp LLM khác.

Cấu hình phục vụ thực tế (`deploy/tei.compose.yml`):

```yaml
tei-gpu:
  image: ghcr.io/huggingface/text-embeddings-inference:cuda-latest
  ports: ["8080:80"]
  command:
    - --model-id=dung6903/agentrag-embed-v1
    - --pooling=mean          # bắt buộc khớp với cách huấn luyện
    - --max-batch-tokens=16384
```

Hai chi tiết vận hành đáng chú ý: (1) phải dùng ảnh `cuda-latest` (bản biên dịch PTX đa kiến trúc) vì GPU RTX 50xx thuộc kiến trúc Blackwell `sm_120`, còn các ảnh ghim phiên bản `:1.5`/`:1.7` chỉ hỗ trợ tới `sm_80` và không chạy được; (2) có hồ sơ `cpu` dự phòng (`cpu-1.5` phục vụ bge-m3) cho máy không có GPU. Phía ứng dụng, `EmbeddingService` bọc thêm một bộ đệm TTL theo khóa SHA-256 của văn bản (2.048 mục, TTL 600 giây, chỉ áp cho lô ≤ 8) — các đường nóng như viết lại truy vấn HyDE hay truy vấn con lặp lại trong vòng decide không phải mã hóa lại cùng một chuỗi.

## 3.5. Mô hình reranker

### 3.5.1. Vì sao cross-encoder chính xác hơn bi-encoder

Bi-encoder (mục 3.4.1) mã hóa câu hỏi và văn bản *độc lập* rồi mới so vector — mô hình chưa từng "nhìn thấy" hai văn bản cạnh nhau, nên chỉ nắm được sự tương đồng ngữ nghĩa tổng quát, dễ nhầm giữa "liều dùng cho người lớn" và "liều dùng cho trẻ em" (hai đoạn rất giống nhau về từ vựng và chủ đề). **Cross-encoder** (bộ mã hóa chéo) khắc phục đúng điểm này: nó ghép câu hỏi và văn bản thành *một* chuỗi đầu vào duy nhất đi qua Transformer, để cơ chế attention so từng token của câu hỏi với từng token của văn bản, rồi xuất một điểm liên quan duy nhất. Độ chính xác cao hơn hẳn, nhưng cái giá là không thể tính trước: mỗi cặp (câu hỏi, văn bản) là một lần suy luận riêng, nên không thể quét cả kho hàng chục nghìn đoạn.

| Tiêu chí | Bi-encoder (embedding) | Cross-encoder (reranker) |
|---|---|---|
| Đầu vào | Câu hỏi và văn bản mã hóa riêng | Cặp (câu hỏi, văn bản) ghép chung |
| Attention giữa hai văn bản | Không | Có, ở mọi tầng |
| Tính trước cho kho tài liệu | Được (index một lần) | Không được |
| Chi phí lúc truy vấn | 1 lần mã hóa + tìm kNN | N lần suy luận cho N ứng viên |
| Độ chính xác xếp hạng | Trung bình | Cao |
| Vai trò trong hệ thống | Giai đoạn 1: quét rộng cả kho | Giai đoạn 2: tinh lọc top ứng viên |

Hệ thống vì vậy dùng kiến trúc truy xuất **hai giai đoạn** kinh điển: truy xuất lai (BM25 + kNN + RRF) lấy nhanh danh sách ứng viên, rồi cross-encoder chấm lại chính xác `RETRIEVAL_RERANK_TOP_N = 20` ứng viên đầu — kết hợp tốc độ của bi-encoder với độ chính xác của cross-encoder.

### 3.5.2. BAAI/bge-reranker-v2-m3 chạy local trên GPU

Mô hình reranker được dùng là **`BAAI/bge-reranker-v2-m3`** — cross-encoder đa ngôn ngữ cùng họ với bge-m3, hỗ trợ tốt tiếng Việt. Nó được nạp qua lớp `CrossEncoder` của thư viện `sentence-transformers` và chạy trực tiếp trên GPU của máy chủ (backend `local_cross_encoder` — giá trị mặc định của `RETRIEVAL_RERANK_BACKEND` trong `config.py`). Điểm thô (logit) của mô hình được đưa qua hàm **sigmoid** để chuẩn về khoảng [0, 1] và gắn lên từng kết quả dưới tên `rerank_score` (mã nguồn `src/agentrag/retrieval/reranker.py`). Nhờ sigmoid, điểm có ngữ nghĩa xác suất ổn định giữa các truy vấn — điều kiện để đặt được một ngưỡng tuyệt đối.

### 3.5.3. Vai trò của rerank_score với an toàn: sàn từ chối trả lời (abstain floor)

Trong miền y tế, trả lời sai nguy hiểm hơn không trả lời. Hệ thống cài một cơ chế an toàn gọi là **sàn liên quan** (relevance floor): nếu điểm `rerank_score` cao nhất trong ngữ cảnh truy xuất được vẫn thấp hơn ngưỡng `RETRIEVAL_RELEVANCE_FLOOR = 0.55`, mô hình được chỉ thị **từ chối trả lời** (abstain) với thông điệp "Tài liệu hiện có không có thông tin…" thay vì bịa ra câu trả lời từ tri thức nội tại. Ngưỡng 0,55 là giá trị đã hiệu chuẩn lại từ 0,6 sau khi phát hiện hiện tượng từ chối nhầm (false abstention) chập chờn trên các câu hỏi hợp lệ có ngữ cảnh mỏng.

Đây chính là lý do kiến trúc buộc reranker phải chạy local: theo tài liệu vận hành của dự án (`docs/HOME-RUN.md`), `local_cross_encoder` là **backend duy nhất phát ra `rerank_score`** — backend thay thế `llm_chat` (nhờ LLM chấm điểm qua chat) không cho điểm số định lượng ổn định nên không đặt ngưỡng được. Ứng dụng thậm chí từ chối khởi động nếu backend `local_cross_encoder` được cấu hình kèm một tên mô hình API — một ràng buộc chủ động để cơ chế an toàn không bị vô hiệu do cấu hình sai. Bài kiểm chứng tiêu chuẩn của dự án: hỏi về một loại thuốc không tồn tại ("Thuốc Zxylopraxin-9 dùng để làm gì?") — hệ thống phải từ chối và không trích dẫn gì; nếu nó trả lời trôi chảy, cấu hình reranker đang sai.

Việc chạy local trên GPU còn cho hai lợi ích phụ: chi phí bằng không cho một tác vụ gọi 20 lần suy luận mỗi truy vấn, và nội dung đoạn tài liệu y tế không phải gửi ra dịch vụ ngoài.

## 3.6. LangGraph — điều phối tác tử dạng đồ thị

### 3.6.1. LangGraph là gì

LangGraph (dự án khai báo `langgraph>=0.2.50` cùng `langchain-core>=0.3.20`) là thư viện điều phối tác tử của hệ sinh thái LangChain, mô hình hóa luồng xử lý thành một **đồ thị trạng thái** (StateGraph): mỗi **node** (nút) là một hàm nhận trạng thái hiện tại và trả về phần cập nhật trạng thái; các **edge** (cạnh) nối thứ tự thực thi; và **conditional edge** (cạnh có điều kiện) chọn nút kế tiếp lúc chạy dựa trên nội dung trạng thái. Khác biệt bản chất so với một pipeline tuyến tính là đồ thị **cho phép chu trình**: luồng có thể quay lại một nút đã đi qua — đúng bản chất của tác tử "suy nghĩ → hành động → quan sát → suy nghĩ tiếp".

### 3.6.2. Ứng dụng trong hệ thống

Luồng chat của VITAL được định nghĩa trong `src/agentrag/agent/graph_service.py` như một `StateGraph(ChatState)` với 13 nút thực tế:

```python
g = StateGraph(ChatState)
g.add_node("validate", validate)            # kiểm tra đầu vào
g.add_node("memory", memory)                # nạp bộ nhớ hội thoại
g.add_node("chitchat_check", chitchat_check)  # phân loại chitchat/nghiệp vụ
g.add_node("chitchat_answer", chitchat_answer)
g.add_node("semantic_plan", semantic_plan)  # lập kế hoạch ngữ nghĩa
g.add_node("bootstrap", bootstrap)
g.add_node("decide", decide)                # tác tử quyết định bước kế
g.add_node("tool_exec", tool_exec)          # thực thi công cụ truy xuất
g.add_node("assemble", assemble)            # lắp ráp ngữ cảnh
g.add_node("answer", answer_node)           # sinh câu trả lời có trích dẫn
g.add_node("critique", critique)            # tự phê bình câu trả lời
g.add_node("corrective_retrieve", corrective_retrieve)  # truy xuất sửa sai
g.add_node("ground", ground)                # kiểm tra tính có căn cứ
```

Ba cạnh có điều kiện tạo nên hành vi tác tử: `_route_chitchat` rẽ nhánh sớm giữa trả lời xã giao và đi vào pipeline RAG đầy đủ (tiết kiệm toàn bộ chi phí truy xuất cho câu chào hỏi); `_route_decide` tạo **vòng lặp decide → tool_exec → decide** — tác tử gọi công cụ truy xuất nhiều lượt cho tới khi tự đánh giá đã đủ thông tin; `_route_critique` tạo **vòng tự sửa** kiểu CRAG (Corrective RAG): nếu bước phê bình thấy câu trả lời chưa đạt, luồng quay về `corrective_retrieve` để truy xuất bổ sung thay vì trả kết quả kém.

Đồ thị được biên dịch kèm **checkpointer** — `InMemorySaver` của LangGraph (`g.compile(checkpointer=_CHECKPOINTER)`): sau mỗi nút, trạng thái được chụp lại theo khóa hội thoại (thread), nhờ đó các lượt hỏi trong cùng hội thoại nối tiếp nhau đúng ngữ cảnh và luồng bị ngắt có thể nối lại từ điểm dừng.

### 3.6.3. So sánh với LangChain

| Tiêu chí | LangChain (chain/AgentExecutor) | LangGraph |
|---|---|---|
| Mô hình luồng | Chuỗi tuyến tính; vòng lặp ẩn trong AgentExecutor, khó can thiệp | Đồ thị tường minh, chu trình là công dân hạng nhất |
| Trạng thái | Ngầm định, truyền qua tham số | Schema trạng thái có kiểu (`ChatState`), mọi nút đọc/ghi một cấu trúc thống nhất |
| Rẽ nhánh có điều kiện | Phải tự viết ngoài chain | `add_conditional_edges` khai báo |
| Checkpoint/nối lại | Không có sẵn | Checkpointer tích hợp |
| Khả năng gỡ lỗi | Khó truy vết luồng ẩn | Từng nút là hàm thuần, quan sát được từng bước |

Hệ thống chỉ dùng `langchain-core` (các kiểu cơ sở mà LangGraph phụ thuộc) chứ không dùng các chain/agent dựng sẵn của LangChain: các luồng có vòng tự sửa và quyết định nhiều bước như trên nếu viết bằng LangChain sẽ phải lách qua các lớp trừu tượng đóng kín, còn với LangGraph chúng là chính hình dạng của đồ thị.

## 3.7. Vision và audio — xử lý tài liệu đa phương thức

Tài liệu y tế thực tế không chỉ là văn bản: PDF scan, hình vẽ giải phẫu, bảng biểu chụp ảnh, tệp ghi âm. Hệ thống trang bị một cụm công nghệ để đưa tất cả về văn bản và vector đánh chỉ mục được.

### 3.7.1. Vision LLM — mô tả ảnh y tế bằng ngôn ngữ

Với hình ảnh trong tài liệu, hệ thống dùng một **mô hình ngôn ngữ thị giác** (Vision LLM) sinh mô tả văn bản cho ảnh; đoạn mô tả này được đánh chỉ mục như văn bản thường, nhờ đó nội dung hình vẽ trở nên "tìm kiếm được" bằng câu hỏi tiếng Việt. Cấu hình qua bộ ba `VISION_PROVIDER` / `VISION_MODEL` / `VISION_BASE_URL`, hỗ trợ ba provider `openai | gemini | ollama` với các mô hình ví dụ trong cấu hình là `gpt-4o`, `gemini-1.5-flash`, `llava:13b` (LLaVA chạy local qua Ollama). Tham số `VISION_TIMEOUT_SECONDS = 180` được nới rộng chủ động để chịu được thời gian nạp nguội (cold start) của LLaVA trên GPU local. Gateway cung cấp cả `vision_response_batch` — gửi N ảnh trong một lời gọi để tiết kiệm hạn mức số yêu cầu/phút của API. Khi `VISION_PROVIDER` không được đặt, `VisionService.enabled = False` và pipeline tự động bỏ qua khâu ảnh (ingestion chỉ-văn-bản) — thiết kế suy giảm nhẹ nhàng (graceful degradation).

### 3.7.2. PyMuPDF và Tesseract OCR — phân tích PDF

**PyMuPDF** (`pymupdf>=1.24.0`, tên khác là fitz) là thư viện Python bọc engine MuPDF, dùng để mở PDF, trích lớp văn bản số và tách các hình ảnh nhúng kèm tọa độ. Với các trang PDF dạng scan — không có lớp văn bản — pipeline chuyển sang **Tesseract OCR** (qua `pytesseract>=0.3.13`): engine nhận dạng ký tự quang học mã nguồn mở của Google, có gói ngôn ngữ tiếng Việt. Sự phối hợp nằm trong `src/agentrag/ingestion/parsers/pdf_parser.py`: văn bản số được ưu tiên (nhanh, chính xác tuyệt đối), OCR chỉ là đường dự phòng cho ảnh scan. Các định dạng văn phòng khác đi qua `markitdown[all]` (bộ chuyển đổi tài liệu → Markdown của Microsoft) và `openpyxl`/`xlrd` cho bảng tính Excel.

### 3.7.3. CLIP — embedding thị giác đa ngôn ngữ

Bên cạnh mô tả ảnh bằng Vision LLM, hệ thống còn tính **embedding thị giác** cho ảnh bằng mô hình `sentence-transformers/clip-ViT-B-32-multilingual-v1` (khai báo tại `VISUAL_EMBEDDING_MODEL`, đầu ra 512 chiều). CLIP (Contrastive Language-Image Pre-training) của OpenAI học một không gian vector *chung* cho ảnh và văn bản: ảnh và câu mô tả nó có vector gần nhau. Biến thể multilingual thay tháp văn bản bằng bộ mã hóa đa ngôn ngữ, nên một truy vấn tiếng Việt có thể so trực tiếp với vector ảnh — mở đường cho tìm kiếm ảnh theo ngôn ngữ tự nhiên mà không phụ thuộc hoàn toàn vào chất lượng đoạn mô tả sinh bởi Vision LLM.

### 3.7.4. faster-whisper — nhận dạng tiếng nói

Tệp âm thanh (mp3/wav/m4a) được chuyển thành văn bản bằng **faster-whisper** (`>=1.0.3`) — bản cài đặt lại mô hình Whisper của OpenAI trên engine suy luận CTranslate2, nhanh hơn đáng kể bản gốc PyTorch với cùng độ chính xác và hỗ trợ lượng tử hóa (int8/float16) để giảm bộ nhớ. Cấu hình thực tế trong `config.py`: `AUDIO_WHISPER_MODEL = "small"` (cân bằng chất lượng/tốc độ cho tiếng Việt), `AUDIO_WHISPER_DEVICE = "auto"` (tự chọn GPU nếu có), `AUDIO_WHISPER_COMPUTE_TYPE = "auto"`, `AUDIO_WHISPER_LANGUAGE = None` (tự phát hiện ngôn ngữ, có thể ghim `"vi"`), `AUDIO_WHISPER_BEAM_SIZE = 5` (tìm kiếm chùm rộng 5 khi giải mã). Văn bản phiên âm sau đó đi vào pipeline cắt đoạn — embedding — đánh chỉ mục như tài liệu thường (`src/agentrag/ingestion/parsers/audio_parser.py`).

## 3.8. Quan sát hệ thống (observability)

Một hệ thống RAG-tác tử thực hiện hàng chục lời gọi mô hình cho mỗi lượt hỏi đáp; không có công cụ quan sát thì việc trả lời các câu hỏi vận hành cơ bản — "lượt chat này chậm ở bước nào?", "tháng này tốn bao nhiêu tiền API?", "vì sao hệ thống từ chối câu hỏi hợp lệ này?" — là bất khả thi. Hệ thống dùng hai cơ chế độc lập.

### 3.8.1. Langfuse — truy vết LLM với cổng chặn dữ liệu nhạy cảm

**Langfuse** là nền tảng quan sát LLM mã nguồn mở, tự vận hành (self-hosted) trong compose của dự án: ảnh `langfuse/langfuse:2` kèm một PostgreSQL riêng, giao diện tại cổng 3002, thuộc profile `observability`. Dự án chủ động ghim **phiên bản 2** (`langfuse>=2.0.0,<3.0.0` trong `pyproject.toml`): SDK v3 chuyển sang giao thức nhận OTEL và đòi hỏi máy chủ v3 với ClickHouse + Redis + worker riêng — quá nặng so với nhu cầu một container của đề tài. Việc khởi tạo hoàn toàn không cần thao tác giao diện nhờ cơ chế bootstrap headless: các biến `LANGFUSE_INIT_*` trong compose tự tạo tổ chức/dự án/khóa API ngay lần chạy đầu.

Mô hình dữ liệu của Langfuse gồm **trace** (dấu vết — toàn bộ một lượt `/chat` là một trace) chứa cây các **span** (nhịp — mỗi bước con: từng nút LangGraph, từng lời gọi LLM, từng truy vấn truy xuất, kèm thời điểm và độ trễ). Nhìn vào một trace, người vận hành thấy chính xác lượt chat đi qua những nút nào, nút nào chậm, mô hình nào được gọi. Phản hồi 👍/👎 của người dùng trên giao diện chat được ghi ngược vào trace tương ứng dưới dạng score `user_feedback` — nối liền trải nghiệm người dùng với dữ liệu chẩn đoán, đồng thời là nguồn khai thác dữ liệu huấn luyện ưu tiên (mục 3.10).

Điểm thiết kế quan trọng nhất với miền y tế là **cổng chặn nội dung**: `OBSERVABILITY_CAPTURE_CONTENT = false` (mặc định và là giá trị vận hành). Khi cờ này tắt, trace chỉ chứa *cấu trúc* — tên bước, mô hình, thời gian, số token — còn **văn bản** câu hỏi/câu trả lời y tế (có thể chứa thông tin sức khỏe cá nhân — PHI, Protected Health Information) không bao giờ được gửi sang kho trace. Đây là cách dung hòa giữa nhu cầu quan sát và nguyên tắc tối thiểu hóa dữ liệu nhạy cảm.

### 3.8.2. Sổ cái chi phí (cost ledger)

Song song với Langfuse, module `src/agentrag/observability/cost.py` duy trì một **sổ cái chi phí** độc lập, gọn nhẹ: mỗi lời gọi LLM thành công (từ `AgentLLM` và `LLMGateway`) ghi một bản ghi {task, model, latency, tokens, USD} vào Redis Stream `agentrag:llm:calls:v1` trên Valkey (giới hạn `MAXLEN ~ 5000` bản ghi gần nhất) — nhờ backend stream, số liệu sống qua khởi động lại và gộp được giữa 4 worker Gunicorn; khi Valkey không với tới được, module lùi về một deque trong tiến trình để không bao giờ làm vỡ đường nghiệp vụ. Số token lấy từ trường `usage` của nhà cung cấp khi có, ngược lại ước lượng theo mật độ ký tự (~4 ký tự/token với ASCII, ~1,8 với tiếng Việt/CJK); thành tiền tính theo bảng giá nội bộ từng mô hình. Kết quả tổng hợp theo từng task và từng model (số lời gọi, token, USD, độ trễ p50/p95) được phơi qua các endpoint `GET /metrics/cost`, `GET /metrics/cost/recent` và hiển thị trên dashboard `http://localhost:3000/cost` của frontend. Toàn bộ cơ chế nằm sau cờ `LLM_COST_TRACKING_ENABLED` và thuần best-effort: lỗi ghi sổ được nuốt sau khi log, không bao giờ chặn câu trả lời cho người dùng. Ngoài ra compose còn kèm tùy chọn **Arize Phoenix** (ảnh `arizephoenix/phoenix`, UI cổng 6006, thu OTLP cổng 4317) — một giao diện trace/eval cục bộ chạy song song Langfuse phục vụ phân tích thí nghiệm.

## 3.9. Hạ tầng triển khai

### 3.9.1. Docker multi-stage — thu gọn ảnh từ 25,7 GB xuống ~9 GB

Ứng dụng được đóng gói bằng **Dockerfile hai giai đoạn** (multi-stage build). Giai đoạn `builder` dựa trên `python:3.11-slim`, cài các gói biên dịch (`build-essential`, `libpq-dev`, `libjpeg-dev`…) và dùng `uv` 0.5 dựng môi trường ảo theo tệp khóa (`uv sync --frozen`). Giai đoạn `runtime` cũng từ `python:3.11-slim` nhưng chỉ cài thư viện *chạy* (`curl` cho healthcheck, `libpq5`, `libjpeg62-turbo`, `zlib1g`) rồi **sao chép nguyên** thư mục `.venv` từ builder — không mang theo trình biên dịch, header hay chính `uv`. Thứ tự layer được sắp có chủ đích: venv (~8 GB, ổn định) nằm trước, mã nguồn (nhỏ, đổi thường xuyên) nằm sau, nên các lần build lại chỉ tốn layer mã nguồn. Kết quả thực tế ghi trong lịch sử dự án: ảnh API giảm từ **25,7 GB xuống ~9 GB**. Phần lớn dung lượng còn lại là PyTorch bản CUDA — được **giữ lại có chủ đích** để ảnh sẵn sàng GPU (reranker cross-encoder), dù các container api/worker hiện chạy torch trên CPU. Ảnh chạy Gunicorn 4 worker UvicornWorker như nêu ở mục 3.1.2; cùng một ảnh dùng cho cả container `api` lẫn `worker` (chỉ khác lệnh khởi động).

### 3.9.2. docker-compose với hồ sơ dịch vụ (profiles)

Toàn bộ hệ thống được mô tả khai báo trong `docker-compose.yml`, chia theo **profile** để bật đúng phần cần thiết:

| Profile | Dịch vụ | Vai trò |
|---|---|---|
| (mặc định) | postgres, elasticsearch, valkey, tei | Hạ tầng lõi + máy chủ embedding |
| `local-llm` | ollama | LLM local (llama3.2:3b, qwen2.5, llava) |
| `app` | api, worker, frontend | Ứng dụng (ảnh `licht693/agentrag-api`, `licht693/agentrag-frontend`) |
| `observability` | langfuse, langfuse-db, phoenix | Truy vết và phân tích |
| `edge` | nginx (`nginx:1.27-alpine`) | Reverse proxy cổng 80 |

Chế độ mặc định là "host-GPU": TEI và Ollama chạy trên máy chủ, các container ứng dụng với tới chúng qua cầu `host.docker.internal` (khai `extra_hosts: host-gateway`). Tệp phủ `docker-compose.fullstack.yml` chuyển sang chế độ "tất cả trong Docker" chỉ bằng cách ghi đè hai biến (`EMBEDDING_BASE_URL=http://tei:80/v1/`, `OLLAMA_BASE_URL=http://ollama:11434/v1/`). Mọi dịch vụ có `healthcheck` và `depends_on ... condition: service_healthy`, nên `docker compose up` tự sắp thứ tự khởi động đúng.

### 3.9.3. GitHub Actions — tích hợp liên tục

Hai workflow trong `.github/workflows/`:

- **`ci.yml`** chạy trên mọi pull request và mọi lần đẩy lên nhánh `master`: dựng hai service container **`pgvector/pgvector:pg16`** (PostgreSQL kèm phần mở rộng pgvector mà lược đồ yêu cầu) và **`elasticsearch:8.15.0`** — đúng phiên bản sản xuất, rồi chạy cổng kiểm thử nhanh `make test-fast` (pytest, các lời gọi LLM/embedding được mock nên CI không cần khóa API) cùng bước lint `ruff` không chặn. Khối `concurrency` với `cancel-in-progress: true` hủy lần chạy cũ khi có commit mới cùng nhánh, tiết kiệm tài nguyên CI.
- **`docker-publish.yml`** build và đẩy hai ảnh `licht693/agentrag-api` và `licht693/agentrag-frontend` lên Docker Hub bằng Buildx, với bước "detect changed build inputs" **chặn theo đường dẫn**: chỉ ảnh nào có đầu vào build thay đổi mới được build lại — tránh build lại ảnh API ~9 GB khi chỉ sửa frontend.

### 3.9.4. Môi trường phần cứng: WSL2 và RTX 5060 Ti 16 GB

Hệ thống được phát triển và vận hành trên **WSL2** (Windows Subsystem for Linux 2) — máy ảo Linux tích hợp trong Windows, cho phép dùng nguyên ngăn xếp Docker/CUDA của Linux trên máy trạm Windows. GPU là **NVIDIA RTX 5060 Ti 16 GB** (kiến trúc Blackwell), truyền vào container qua NVIDIA Container Toolkit; compose của dự án ghi chú sẵn phương án dự phòng CDI (`nvidia-ctk cdi generate`) cho lỗi segfault đã biết của toolkit trên WSL2. Kiến trúc Blackwell (`sm_120`) mới đến mức ảnh TEI ghim phiên bản chưa hỗ trợ — phải dùng ảnh `cuda-latest` biên dịch PTX đa kiến trúc (mục 3.4.4). Ngân sách 16 GB VRAM là ràng buộc thiết kế xuyên suốt: đủ cho ba mô hình Ollama thường trú đồng thời (`qwen2.5:7b` + `llava:7b` + `llama3.2:3b`), đủ cho TEI và reranker, đủ cho các phiên huấn luyện ở mục 3.10 — nhưng là lằn ranh loại trừ phương án DPO 7B.

## 3.10. Công cụ huấn luyện

Chất lượng của hệ thống không chỉ đến từ lắp ghép mô hình có sẵn mà từ việc **tinh chỉnh mô hình trên dữ liệu của chính hệ thống**. Hai nhóm công cụ phục vụ việc này.

### 3.10.1. sentence-transformers — tinh chỉnh embedding và reranker

**sentence-transformers** (`>=3.0.0`) là thư viện tiêu chuẩn để huấn luyện và suy luận các mô hình embedding câu; bản 3.x xây trên HF Trainer nên cần thêm hai gói `datasets` và `accelerate` — được khai báo thành extra riêng `finetune` trong `pyproject.toml` (`uv sync --extra finetune`).

**Tinh chỉnh embedding** (`scripts/finetune_embedding.py`): mô hình nền mặc định `intfloat/multilingual-e5-base` được huấn luyện bằng **MultipleNegativesRankingLoss** — hàm mất mát tương phản coi mỗi bộ ba (truy vấn, đoạn đúng, đoạn nhiễu khó) là một mẫu, đồng thời tận dụng các đoạn đúng của những truy vấn khác trong cùng lô làm mẫu nhiễu bổ sung (in-batch negatives), nhờ đó một lô 16 mẫu tạo ra hàng trăm cặp tương phản. Dữ liệu 5.300 bộ ba được khai thác tự động từ chính hệ thống (`make mine-pairs`): sinh câu hỏi tổng hợp từ các đoạn tài liệu, kết hợp phản hồi 👍/👎 thật của người dùng, và đào mẫu nhiễu khó (hard negatives) từ các kết quả truy xuất gần đúng. Yêu cầu phần cứng chỉ ~5 GB VRAM với e5-base ở batch 16 (~10 GB nếu nền là bge-m3) — vừa vặn GPU 16 GB. Quy trình có **cổng định lượng**: `scripts/eval_retrieval.py` so mô hình ứng viên với mô hình nền trên tập kiểm thử 10% tách riêng; chỉ thăng cấp khi có cải thiện (kết quả thực tế recall@10 +0,20 như mục 3.4.3).

**Tinh chỉnh reranker** (`scripts/finetune_reranker.py`): cùng thư viện, lớp `CrossEncoder` với `num_labels=1, max_length=512`, mô hình nền `BAAI/bge-reranker-v2-m3`, huấn luyện trên nhãn liên quan nhị phân từ cùng nguồn dữ liệu — bước tùy chọn sau khi embedding đã thăng cấp.

### 3.10.2. LoRA/QLoRA và tinh chỉnh ưu tiên KTO/ORPO cho LLM

Tinh chỉnh toàn bộ trọng số một LLM hàng tỷ tham số vượt xa 16 GB VRAM. **LoRA** (Low-Rank Adaptation — thích ứng hạng thấp) giải quyết bằng cách đóng băng trọng số gốc và chỉ huấn luyện các ma trận hiệu chỉnh hạng thấp gắn thêm vào từng tầng attention; **QLoRA** đi xa hơn bằng cách nạp trọng số gốc đã lượng tử hóa 4-bit. Script `scripts/finetune_dpo.py` của dự án dùng **Unsloth** (`unsloth[cu121-torch240]`) — thư viện tối ưu bộ nhớ/tốc độ cho LoRA — với cấu hình thực tế: adapter hạng `r=16` (`lora_alpha = 2r`), gradient checkpointing kiểu unsloth, mô hình nền mặc định `unsloth/Qwen2.5-3B-Instruct`, độ dài chuỗi 2048; sau huấn luyện, adapter được trộn ngược vào trọng số và xuất safetensors 16-bit, rồi chuyển sang định dạng Ollama (`scripts/convert_to_ollama.sh`) để cắm vào `LLM_TASK_MODEL_MAP`.

Về thuật toán học ưu tiên (preference tuning), dự án chọn **KTO** (Kahneman-Tversky Optimization) và **ORPO** (Odds Ratio Preference Optimization) qua thư viện `trl` của Hugging Face, thay vì DPO kinh điển, vì hai lý do khớp trực tiếp với ràng buộc của đề tài:

1. **Dạng dữ liệu.** DPO/ORPO cần cặp so sánh {prompt, câu trả lời tốt, câu trả lời xấu}; KTO chỉ cần nhãn nhị phân tốt/xấu trên từng câu trả lời — đúng dạng phản hồi 👍/👎 mà giao diện chat thu về tự nhiên (`scripts/mine_preference.py` xuất được cả hai định dạng; kế hoạch kích hoạt khi tích lũy ≥ 500 phản hồi).
2. **Bộ nhớ.** DPO phải giữ đồng thời cặp chosen+rejected *và* một mô hình tham chiếu đông cứng — với mô hình 7B, tổng nhu cầu chạm trần 16 GB và gây tràn bộ nhớ (tài liệu vận hành của dự án ghi rõ "không chạy 7B DPO trên 16 GB"). KTO và ORPO đều **không cần mô hình tham chiếu** (reference-free), nên phiên QLoRA 3B chỉ tốn ~8–11 GB — nằm an toàn trong ngân sách VRAM.

Chuỗi công cụ này khép kín một vòng cải tiến liên tục hoàn toàn tại chỗ: phản hồi người dùng (Langfuse/PostgreSQL) → khai thác dữ liệu ưu tiên → tinh chỉnh trên GPU cục bộ → phục vụ qua Ollama/TEI → đo lại bằng cổng đánh giá — không dữ liệu y tế nào rời khỏi hạ tầng tự vận hành trong suốt vòng đời huấn luyện.

## Kết chương

Ngăn xếp công nghệ của VITAL được lựa chọn theo ba nguyên tắc nhất quán. **Một giao thức, nhiều nhà cung cấp**: chuẩn OpenAI-compatible thống nhất mọi điểm gọi mô hình (Ollama, DeepSeek, Gemini, Claude, TEI), cho phép định tuyến từng tác vụ đến đúng mô hình theo `LLM_TASK_MODEL_MAP` mà không sửa mã nghiệp vụ. **An toàn và riêng tư là ràng buộc kiến trúc, không phải tùy chọn**: reranker buộc chạy local để có `rerank_score` định lượng nuôi cơ chế từ chối trả lời; cổng `OBSERVABILITY_CAPTURE_CONTENT=false` giữ nội dung y tế ngoài kho trace; các tác vụ điều phối tần suất cao chạy trên mô hình local. **Vừa vặn tài nguyên thực tế**: mọi lựa chọn — từ Langfuse v2 một container, ảnh Docker tinh gọn 9 GB, đến cặp thuật toán KTO/ORPO không cần mô hình tham chiếu — đều được cân đối cho một máy trạm WSL2 với GPU 16 GB VRAM. Chương 4 tiếp theo sẽ trình bày cách các công nghệ này được lắp ráp thành kiến trúc tổng thể và các luồng xử lý cụ thể của hệ thống.
