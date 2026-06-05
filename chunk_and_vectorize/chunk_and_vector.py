import glob
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader.load_documents import load

# Define the base directory relative to this script
# explore_knowledge_base/get_knowledge_base_info.py -> explore_knowledge_base -> rag_demo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = str(PROJECT_ROOT / "knowledge-base" / "**" / "*.md")
DB_NAME = "vector_db"


def chunk_knowledge_base(knowledge_base: str):
    """
    Divides the provided knowledge base into smaller chunks for easier processing and analysis.

    The function processes text documents by splitting them into manageable segments using
    the RecursiveCharacterTextSplitter. The resulting chunks are sub-parts of the original
    documents, ensuring overlaps to maintain context between chunks.

    :param knowledge_base: The path or name of the knowledge base to be processed.
    :type knowledge_base: str
    :return: A list of document chunks derived from the knowledge base.
    :rtype: list
    """
    kb_folder = PROJECT_ROOT / "knowledge-base"
    folders = glob.glob(str(kb_folder / "*"))
    documents = load(folders)

    # Divide into chunks using the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # returns the chunks
    print(text_splitter.split_documents(documents)[0].page_content[0:100])
    return text_splitter.split_documents(documents)

def vectorize_chunks(chunks, embeddings):
    """
    Converts a list of document chunks into a vector representation using the specified embeddings.

    :param chunks: List of document chunks to be vectorized.
    :type chunks: list
    :param embeddings: Embedding model to use for vectorization.
    :type embeddings: Embeddings
    :return: Vectorized representation of the document chunks.
    :rtype: Chroma
    """
    print(f"Vectorizing {len(chunks)} chunks with {embeddings}...")

    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    return vectorstore