import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from chunk_and_vectorize.chunk_and_vector import chunk_knowledge_base, vectorize_chunks
from explore_knowledge_base.get_knowledge_base_info import load_knowledge_base, get_token_count, get_character_count
from visualize_vectors.display_scatter_plots import display_2d_scatter, display_3d_scatter

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

MODEL = "gpt-4.1-nano"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_BASE_PATH = str(PROJECT_ROOT / "knowledge-base" / "**" / "*.md")

EMBEDDING_MODEL = [HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                   OpenAIEmbeddings(model="text-embedding-3-small"),
                   OpenAIEmbeddings(model="text-embedding-3-large")]


def view_chunk_info(chunks: list[Document], index: int = 0):
    """
    Generate a formatted string with chunk information.

    :param chunks: List of Document objects representing chunks.
    :param index: Index of the chunk to display.
    :return: Formatted string with chunk information.
    """
    return f"Total of {len(chunks)} chunks: \nchunk {index + 1}:\n\n{chunks[index]}"

def view_vector_store(vectorstore):
    """
    Generate a formatted string with vector store information.

    :param vectorstore: Vector store object.
    :return: Formatted string with vector store information.
    """
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    return f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"



def main():
    entire_knowledge_base = load_knowledge_base(DEFAULT_KNOWLEDGE_BASE_PATH)
    chunks = chunk_knowledge_base(entire_knowledge_base)

    view_chunk = view_chunk_info(chunks)
    vectorstore = vectorize_chunks(chunks, EMBEDDING_MODEL[0])
    vector_visualization = view_vector_store(vectorstore)
    two_d_scatter = display_2d_scatter(vectorstore._collection)
    three_d_scatter = display_3d_scatter(vectorstore._collection)

    print(view_chunk)
    print(vector_visualization)


if __name__ == "__main__":
    main()
