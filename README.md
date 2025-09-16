# MedicalQuestionsAgent (Streamlit + LlamaIndex + Chroma)

Chat with your own local corpus of **medical health documents** using a Streamlit UI.
Under the hood it uses:

* **LlamaIndex** (`VectorStoreIndex`, sentence chunking, post-processors)
* **OpenAI** (LLM + embeddings)
* **ChromaDB** (persistent vector store)
* **SentenceTransformer Reranker** (`BAAI/bge-reranker-base`) for better result ordering

> ⚕️ **Disclaimer:** This project is for educational purposes. It does **not** provide medical advice and is **not** a substitute for professional diagnosis or treatment.

---

## Features

* 🗂️ **Bring your own docs**: drop files into `./data` (supports nested folders)
* 🧩 **Smart chunking** with a `SentenceSplitter` and sliding windows
* 🧠 **OpenAI embeddings + LLM** with configurable model/temperature
* 🗃️ **Persistent** local vector store via Chroma (`./chroma_db`)
* 🎯 **Reranking** with `BAAI/bge-reranker-base` to surface the most relevant passages
* 💬 **Chat UX** with history using `st.session_state`

---

## Quickstart

### 1) Prerequisites

* Python **3.10+** (recommended)
* An OpenAI API key
* Disk space for a local Chroma DB (defaults to `./chroma_db`)

### 2) Clone & install

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Example `requirements.txt` (works well with the shown imports):**

```txt
streamlit>=1.33
openai>=1.30
llama-index>=0.10.40
llama-index-core>=0.10.40
llama-index-llms-openai>=0.1.18
llama-index-embeddings-openai>=0.1.10
llama-index-vector-stores-chroma>=0.1.6
chromadb>=0.5.3
sentence-transformers>=3.0.1
torch>=2.2
```

> If you pin versions later, keep LlamaIndex packages on compatible minor versions to avoid breaking imports like `llama_index.core`.

### 3) Add your documents

Place PDFs / TXTs / docs in:

```
./data
└── (your folders & files)  # recursive loading is enabled
```

### 4) Provide your OpenAI key

You can use **either** method below:

**A) Environment variable**

```bash
export OPENAI_API_KEY="sk-..."   # Windows (PowerShell):  $env:OPENAI_API_KEY="sk-..."
```

**B) Streamlit secrets** (create `.streamlit/secrets.toml`)

```toml
openai_key = "sk-..."
```

> The code prefers `OPENAI_API_KEY` if present, otherwise it reads `st.secrets.openai_key`.

### 5) Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)), then ask your medical questions.

---

## How it works

1. **Ingestion & Chunking**

   * `SimpleDirectoryReader` loads files from `./data` (recursive).
   * `SentenceSplitter` breaks text into \~512-token chunks and attaches a “window” of surrounding context for better answers.

2. **Indexing**

   * **Embeddings**: `OpenAIEmbedding()` converts chunks to vectors.
   * **Vector store**: Chroma is initialized with a persistent collection (`tech16example`) at `./chroma_db`.
   * **Index**: `VectorStoreIndex` is created over docs/nodes with the above components.
   * The entire index is **cached** with `@st.cache_resource`, so restarts are faster once built.

3. **Post-processing & Reranking**

   * `MetadataReplacementPostProcessor` swaps single sentences with their “window” context for more readable answers.
   * `SentenceTransformerRerank(top_n=3, model="BAAI/bge-reranker-base")` reorders retrieved nodes by semantic relevance.

4. **Chat**

   * `index.as_chat_engine(chat_mode="react", similarity_top_k=10, node_postprocessors=[...])`
   * Maintains chat history in `st.session_state.messages`.

---

## Configuration

You can tweak these directly in `app.py`:

* **Model & tone**

  * `OpenAI(model="gpt-3.5-turbo", temperature=0.3, system_prompt=...)`
  * Swap to `gpt-4o-mini` / `gpt-4o` etc. if you have access.

* **Reranker**

  * `model="BAAI/bge-reranker-base"`, `top_n=3`
  * Requires `sentence-transformers` (and `torch`) to download and run the model.

* **Retriever breadth**

  * `similarity_top_k=10` controls how many candidate chunks are pulled before reranking.

* **Chroma persistence**

  * Path is `./chroma_db`; safe to delete the folder if you wish to **reindex from scratch**.

---

## Project structure

```
.
├─ app.py                 # the Streamlit app (your current file)
├─ data/                  # put your source documents here
├─ chroma_db/             # auto-created persistent vector store
├─ .streamlit/
│  └─ secrets.toml        # optional: holds openai_key
└─ requirements.txt
```

---

## Common tasks

**Rebuild the index from scratch**

```bash
rm -rf ./chroma_db
streamlit run app.py
```

**Change the OpenAI model**

* Update `OpenAI(model=...)` in the `ServiceContext.from_defaults(...)` block.

**Control chunk size**

* Adjust `SentenceSplitter.from_defaults(chunk_size=512, ...)`.

---

## Troubleshooting

* **`openai.AuthenticationError` or “No API key provided”**

  * Ensure `OPENAI_API_KEY` is exported or `.streamlit/secrets.toml` contains `openai_key`.

* **`ModuleNotFoundError: llama_index...`**

  * Verify version pins match the imports (`llama_index.core`, `llama_index.llms.openai`, etc.).

* **Reranker download too slow / blocked**

  * Pre-download the model or set `SentenceTransformerRerank(..., model="/path/to/local/model")`.

* **“CUDA not available” warnings**

  * Totally fine; the reranker will run on CPU (slower but works). Install CUDA-enabled PyTorch if you want GPU.

* **Indexing seems stuck on first run**

  * Large corpora + embeddings + reranker downloads can take time. Subsequent runs use the cached Chroma store.

---

## Security & privacy

* Documents remain **local**; vectors are stored in `./chroma_db`.
* Text chunks are sent to OpenAI for embeddings and LLM responses; review your data governance before using sensitive PHI.
* This app is **not** HIPAA compliant.

---

## Roadmap (nice next steps)

* 🔎 Add source citations in the chat responses
* 🧪 Add evaluation set to measure answer quality
* 🧰 UI controls for `top_k`, temperature, and reranker `top_n`
* 🚀 Dockerfile + one-click deploy (Streamlit Community Cloud, Fly.io, etc.)
* 🧱 Model switcher (OpenAI / Azure / local LLMs)

---

## Acknowledgments

* [LlamaIndex](https://www.llamaindex.ai/)
* [ChromaDB](https://www.trychroma.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [Streamlit](https://streamlit.io/)

---
