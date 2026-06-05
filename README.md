---
title: Insurellm RAG Explorer
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.16.0"
app_file: app.py
pinned: false
---

# Insurellm RAG Knowledge Base Explorer

An interactive Retrieval-Augmented Generation (RAG) demo built for portfolio demonstration. Explore a fictional insurance company knowledge base through document preview, chunk inspection, embedding visualization, and semantic Q&A.

## Features

- **Knowledge base browser** — preview any of 76 markdown documents across products, contracts, employees, and company info
- **Embedding model selection** — HuggingFace `all-MiniLM-L6-v2` (local) or OpenAI `text-embedding-3-small` / `text-embedding-3-large`
- **Vector visualization** — interactive 2D/3D t-SNE scatter plots colored by document type
- **Chunk inspector** — browse individual text chunks with source metadata
- **RAG Q&A** — ask questions grounded in retrieved context with source citations

## Tech Stack

Built with: **Python** · **Gradio** · **ChromaDB** · **HuggingFace sentence-transformers** · **OpenAI GPT** (Q&A) · **Plotly/t-SNE** · **LangChain** document loaders & text splitter

| Layer | Technology |
|-------|------------|
| UI | Gradio 6 |
| Vector store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (default), OpenAI optional |
| Document loading | LangChain Community `DirectoryLoader` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| LLM | OpenAI GPT-4.1-nano (RAG answers) |
| Visualization | Plotly + scikit-learn t-SNE |

## Architecture

```
knowledge-base/*.md
       │
       ├──► Document loader (metadata: doc_type)
       │
       ├──► RecursiveCharacterTextSplitter (1000 / 200)
       │
       ├──► Embedding model → ChromaDB (vector_db/)
       │
       ├──► t-SNE scatter plots (2D / 3D)
       │
       └──► Retriever + LLM (RAG answers)
```

## Local Development

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone <your-repo-url>
cd rag_demo
uv sync
cp .env.example .env   # optional — needed for OpenAI embeddings and RAG Q&A
```

### Run the Gradio app

```bash
uv run python gradio_app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Run the CLI pipeline

```bash
uv run python main.py
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Optional* | OpenAI embeddings and RAG question answering |

\* The default HuggingFace embedding model works without an API key. OpenAI features are disabled until the key is set.

## Deploy to Hugging Face Spaces

1. Create a new Gradio Space on [Hugging Face](https://huggingface.co/new-space)
2. Push this repository (or connect via Git)
3. Set `OPENAI_API_KEY` as a **Space secret** (Settings → Secrets) for RAG Q&A
4. The Space uses `app.py` as the entry point

The app pre-warms the vector store on startup with the local HuggingFace model so visitors see plots immediately.

### Embed on your portfolio site

```html
<iframe
  src="https://<your-username>-insurellm-rag-explorer.hf.space"
  width="100%"
  height="900"
  frameborder="0"
></iframe>
```

## Project Structure

```
rag_demo/
├── app.py                  # Hugging Face Spaces entry point
├── gradio_app.py           # Gradio UI
├── rag_service.py          # Shared business logic
├── main.py                 # CLI entry point
├── knowledge-base/         # Source documents (76 .md files)
├── document_loader/        # LangChain document loading
├── chunk_and_vectorize/    # Chunking + Chroma persistence
├── explore_knowledge_base/ # KB stats and file utilities
└── visualize_vectors/      # Plotly t-SNE scatter plots
```

## License

MIT — built as a portfolio project for software engineering demonstration.
