import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chunk_and_vectorize.chunk_and_vector import chunk_knowledge_base, vectorize_chunks
from explore_knowledge_base.get_knowledge_base_info import (
    get_character_count,
    get_doc_type_breakdown,
    get_token_count,
    list_kb_files,
    load_knowledge_base,
    read_kb_file,
)
from visualize_vectors.display_scatter_plots import get_scatter_fig

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_MODEL = "gpt-4.1-nano"
TOKEN_COUNT_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL_OPTIONS = {
    "all-MiniLM-L6-v2 (HuggingFace, local)": "huggingface",
    "text-embedding-3-small (OpenAI)": "openai-small",
    "text-embedding-3-large (OpenAI)": "openai-large",
}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2 (HuggingFace, local)"

KB_FILE_EMPTY_STATE = "Select a file from the dropdown above to preview it here."

PRODUCT_QUESTION_KEYWORDS = (
    "product",
    "products",
    "offer",
    "offers",
    "sell",
    "sells",
    "solution",
    "solutions",
    "service",
    "services",
)


@dataclass
class PipelineState:
    chunks: list[Document] | None = None
    vectorstore: object | None = None
    embedding_model: str | None = None


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def requires_openai_key(model_label: str) -> bool:
    return EMBEDDING_MODEL_OPTIONS[model_label].startswith("openai")


def validate_embedding_model(model_label: str) -> str | None:
    if requires_openai_key(model_label) and not get_openai_api_key():
        return "OPENAI_API_KEY is required for OpenAI embedding models."
    return None


def get_embedding_model(model_label: str):
    model_type = EMBEDDING_MODEL_OPTIONS[model_label]
    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if model_type == "openai-small":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if model_type == "openai-large":
        return OpenAIEmbeddings(model="text-embedding-3-large")
    raise ValueError(f"Unknown embedding model: {model_label}")


def build_pipeline(model_label: str) -> tuple[PipelineState, str | None]:
    error = validate_embedding_model(model_label)
    if error:
        return PipelineState(), error

    chunks = chunk_knowledge_base()
    embeddings = get_embedding_model(model_label)
    vectorstore = vectorize_chunks(chunks, embeddings)

    return PipelineState(
        chunks=chunks,
        vectorstore=vectorstore,
        embedding_model=model_label,
    ), None


def get_overview_stats() -> str:
    knowledge_base = load_knowledge_base()
    stats = [
        "# Knowledge Base Overview",
        "",
        get_character_count(knowledge_base),
    ]
    try:
        stats.append(get_token_count(TOKEN_COUNT_MODEL, knowledge_base))
    except Exception:
        stats.append(
            f"Total tokens for {TOKEN_COUNT_MODEL}: unavailable "
            "(token encoding could not be loaded)"
        )
    stats.extend(["", get_doc_type_breakdown()])
    return "\n".join(stats)


def format_relative_source(source: str) -> str:
    if not source or source == "unknown":
        return "unknown"

    path = Path(source)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        if "knowledge-base" in path.as_posix():
            idx = path.as_posix().index("knowledge-base")
            return path.as_posix()[idx:]
        return path.name


def format_kb_file_preview(relative_path: str | None) -> str:
    if not relative_path:
        return KB_FILE_EMPTY_STATE

    file_data = read_kb_file(relative_path)
    return (
        f"**File:** `{file_data['path']}`  \n"
        f"**Category:** {file_data['category'].title()}  \n"
        f"**Characters:** {file_data['char_count']:,}\n\n"
        f"---\n\n"
        f"{file_data['content']}"
    )


def get_chunk_choices(chunks: list[Document] | None) -> list[tuple[str, int]]:
    if not chunks:
        return []

    choices = []
    for index, chunk in enumerate(chunks):
        relative_source = format_relative_source(chunk.metadata.get("source", "unknown"))
        short_source = Path(relative_source).name if relative_source != "unknown" else "unknown"
        label = f"Chunk {index + 1} — {short_source[:60]}"
        choices.append((label, index))
    return choices


def get_chunk_info(chunks: list[Document] | None, index: int) -> str:
    if not chunks:
        return "Build the vector store to inspect document chunks."

    if index < 0 or index >= len(chunks):
        return "Select a valid chunk to inspect."

    chunk = chunks[index]
    source = format_relative_source(chunk.metadata.get("source", "unknown"))
    doc_type = chunk.metadata.get("doc_type", "unknown")
    return (
        f"**Total chunks:** {len(chunks):,}  \n"
        f"**Selected chunk:** {index + 1} of {len(chunks):,}  \n"
        f"**Source:** `{source}`  \n"
        f"**Document type:** {doc_type}\n\n"
        f"---\n\n"
        f"{chunk.page_content}"
    )


def get_vector_stats(vectorstore) -> str:
    if vectorstore is None:
        return "Build the vector store to view embedding statistics."

    collection = vectorstore._collection
    count = collection.count()
    if count == 0:
        return "The vector store is empty."

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    return (
        f"**Vectors:** {count:,}  \n"
        f"**Dimensions:** {dimensions:,}  \n"
        f"**Storage:** ChromaDB (`vector_db/`)"
    )


def get_plot(vectorstore, dim: str):
    if vectorstore is None:
        return None
    return get_scatter_fig(vectorstore._collection, dim)


def _doc_key(doc: Document) -> tuple[str, str]:
    return (doc.page_content[:200], doc.metadata.get("source", ""))


def _merge_dedupe(docs: list[Document], limit: int = 8) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    merged: list[Document] = []
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def _is_product_question(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in PRODUCT_QUESTION_KEYWORDS)


def _score_map(vectorstore, question: str) -> dict[tuple[str, str], float]:
    scored_hits = vectorstore.similarity_search_with_score(question, k=24)
    scores: dict[tuple[str, str], float] = {}
    for doc, distance in scored_hits:
        scores[_doc_key(doc)] = float(distance)
    return scores


def retrieve_context_docs(vectorstore, question: str) -> list[tuple[Document, float | None]]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.6},
    )
    docs = retriever.invoke(question)

    if _is_product_question(question):
        product_docs = vectorstore.similarity_search(
            question,
            k=6,
            filter={"doc_type": "products"},
        )
        docs = _merge_dedupe(product_docs + docs, limit=8)

    scores = _score_map(vectorstore, question)
    return [(doc, scores.get(_doc_key(doc))) for doc in docs]


def _format_relevance(distance: float | None) -> str:
    if distance is None:
        return "n/a"
    relevance = max(0.0, min(1.0, 1 / (1 + distance)))
    return f"{relevance:.2f}"


def _format_source_block(doc: Document, rank: int, distance: float | None = None) -> str:
    source = format_relative_source(doc.metadata.get("source", "unknown"))
    doc_type = doc.metadata.get("doc_type", "unknown")
    preview = doc.page_content[:400].replace("\n", " ")
    if len(doc.page_content) > 400:
        preview += "..."
    return (
        f"### Source {rank}\n"
        f"- **File:** `{source}`\n"
        f"- **Type:** {doc_type}\n"
        f"- **Relevance:** {_format_relevance(distance)}\n"
        f"- **Excerpt:** {preview}"
    )


def rag_query(vectorstore, question: str) -> tuple[str, str]:
    if vectorstore is None:
        return "Build the vector store before asking questions.", ""
    if not question.strip():
        return "Enter a question to search the knowledge base.", ""

    if not get_openai_api_key():
        return (
            "RAG answers require OPENAI_API_KEY. "
            "Set it in your environment to enable question answering.",
            "",
        )

    ranked_docs = retrieve_context_docs(vectorstore, question)

    context_blocks = []
    for index, (doc, _) in enumerate(ranked_docs, start=1):
        source = format_relative_source(doc.metadata.get("source", "unknown"))
        context_blocks.append(f"[Source {index}: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the provided context from Insurellm, "
                "a fictional insurance software company knowledge base. "
                "When asked about products or offerings, list every distinct product "
                "name found in the context (many end in 'llm', e.g. Carllm, Homellm). "
                "Synthesize information across all provided sources. "
                "Cite source file paths in your answer. "
                "Only say you lack information if the context truly does not contain it.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = (prompt | llm).invoke({"context": context, "question": question})
    answer = response.content

    sources = "\n\n".join(
        _format_source_block(doc, index, distance)
        for index, (doc, distance) in enumerate(ranked_docs, start=1)
    )
    return answer, sources


def prewarm_pipeline(model_label: str = DEFAULT_EMBEDDING_MODEL) -> PipelineState:
    state, error = build_pipeline(model_label)
    if error:
        return PipelineState()
    return state
