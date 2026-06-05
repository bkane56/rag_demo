import glob
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader.load_documents import load


# Define the base directory relative to this script
# explore_knowledge_base/get_knowledge_base_info.py -> explore_knowledge_base -> rag_demo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = str(PROJECT_ROOT / "knowledge-base" / "**" / "*.md")


def load_knowledge_base(path: str = DEFAULT_KB_PATH):
    """
    Loads and concatenates the contents of all Markdown files from a specified file path
    into a single string. The files are located recursively based on the input `path`.

    :param path: A filesystem path pattern to search for Markdown files.
                 Defaults to "knowledge-base/**/*.md".
    :type path: str
    :return: A string containing the concatenated contents of all located Markdown
             files. An empty string is returned if no files are found.
    :rtype: str
    """
    files = glob.glob(path, recursive=True)

    entire_knowledge_base = ""

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            entire_knowledge_base += f.read()
            entire_knowledge_base += "\n\n"
    return entire_knowledge_base

def get_character_count(knowledge_base: str = load_knowledge_base()):
    """
    Calculates the total character count of the knowledge base.

    :param knowledge_base: The knowledge base content to count characters from.
                           Defaults to the loaded knowledge base.
    :type knowledge_base: str
    :return: The total character count of the knowledge base.
    :rtype: int
    """

    return f"Total characters in knowledge base: {len(knowledge_base):,}"


def get_token_count(model: str, knowledge_base: str = load_knowledge_base()):
    """
    Calculates the total token count of the knowledge base for a given model.

    :param model: The model to use for tokenization.
    :type model: str
    :param knowledge_base: The knowledge base content to count tokens from.
                           Defaults to the loaded knowledge base.
    :type knowledge_base: str
    :return: The total token count of the knowledge base for the given model.
    :rtype: str
    """
    encoding = tiktoken.encoding_for_model(model)
    token_count = encoding.encode(knowledge_base)
    return f"Total tokens for {model}: {token_count:,}"
