# 🚀 RAG Observability Pro: Llama 3.2 + Arize Phoenix

A production-hardened Retrieval-Augmented Generation (RAG) pipeline optimized for low-resource environments (T4 GPUs). This project features **Llama 3.2 (3B)**, **Unsloth 4-bit quantization**, and full **OpenTelemetry** instrumentation for real-time auditability.

## 🛠️ System Architecture
To ensure LLM Safety and Hallucination Correction, the pipeline follows a strictly modular data flow:



1. **Ingestion:** Documents are processed via `SentenceSplitter` (8ms-45ms latency).
2. **Retrieval:** Relevant context is fetched from a `SummaryIndex`.
3. **Augmentation:** Context is injected into the prompt via a stabilized `CustomLLM` adapter.
4. **Generation:** Llama 3.2 3B produces the final response with KV-Cache hardening.

## 🛡️ Infrastructure Resilience (Key Engineering Wins)
During development, several critical hardware-level bottlenecks were identified and bypassed to ensure 100% system uptime:
* **KV-Cache Shape Stabilization:** Resolved a persistent `RuntimeError` regarding broadcasting mismatches (`[1, 24, 1, 128]` vs `[1, 24, 52, 128]`) by implementing a cache-free inference bypass (`use_cache=False`).
* **Python 3.12 Compatibility Shield:** Manually patched `pkgutil.ImpImporter` and `torch.int1` references to support legacy OpenTelemetry wrappers in modern environments.

## 📊 Observed Performance Metrics
Data captured during a stress test on **March 31, 2026**:

| Metric | Observed Value |
| :--- | :--- |
| **P50 Latency (Median)** | **13.3s** |
| **P99 Latency (Worst Case)** | **19.2s** |
| **Success Rate** | **100%** |

## 🔍 Observability & Audit Trail
This system utilizes **Arize Phoenix** to provide a complete "X-Ray" view of the AI's reasoning process.

![Phoenix Traces Audit](docs/images/img1.png)
*Above: Successful trace execution for adversarial and logical stress-test queries.*

![Phoenix Traces Audit](docs/images/img2.png)
*Above: Detailed Span Tree showing the precise timing of Retrieval vs. Generation stages.*

## 📂 Project Structure
```text
├── src/
│   ├── __init__.py     # Package marker
│   └── app.py          # Stabilized RAG logic & CustomLLM Adapter
├── .gitignore          # Data Leak Prevention (DLP)
├── README.md           # Technical Audit & Performance Findings
└── requirements.txt    # Frozen dependency versions


## 🚀 Quick Start
1. **Clone the repo:** `git clone https://github.com/ShettyShreyasR/rag-observability-pro.git`
2. **Install dependencies:** `pip install -r requirements.txt` 
   *(Note: Restart your Python session after installation to activate the environment patches)*
3. **Launch Observability:** `python -m phoenix.server.main` (or run `px.launch_app()` in a notebook)
4. **Initialize and Query:**
   ```python
   from src.app import initialize_engine
   from llama_index.core import SummaryIndex, Document

   # 1. Start the hardened engine
   Settings = initialize_engine()

   # 2. Build a simple index
   doc = Document(text="The butterfly effect is a concept in chaos theory.")
   index = SummaryIndex.from_documents([doc])
   query_engine = index.as_query_engine()

   # 3. Query (Traces will automatically appear in Phoenix)
   response = query_engine.query("What is the butterfly effect?")
   print(response)
