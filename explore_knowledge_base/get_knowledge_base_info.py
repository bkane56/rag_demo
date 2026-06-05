import glob
from pathlib import Path

import tiktoken

# Define the base directory relative to this script
# explore_knowledge_base/get_knowledge_base_info.py -> explore_knowledge_base -> rag_demo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_ROOT = PROJECT_ROOT / "knowledge-base"
DEFAULT_KB_PATH = str(KB_ROOT / "**" / "*.md")


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


def list_kb_files() -> list[str]:
    """Return sorted relative paths for all markdown files in the knowledge base."""
    files = glob.glob(DEFAULT_KB_PATH, recursive=True)
    return sorted(str(Path(f).relative_to(PROJECT_ROOT)) for f in files)


def read_kb_file(relative_path: str) -> dict:
    """Read a knowledge-base file and return content with metadata."""
    file_path = PROJECT_ROOT / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {relative_path}")

    content = file_path.read_text(encoding="utf-8")
    parts = Path(relative_path).parts
    category = parts[1] if len(parts) > 1 else "unknown"

    return {
        "path": relative_path,
        "category": category,
        "char_count": len(content),
        "content": content,
    }


def get_doc_type_breakdown() -> str:
    """Return a formatted summary of files per knowledge-base category."""
    counts: dict[str, int] = {}
    for relative_path in list_kb_files():
        category = Path(relative_path).parts[1] if len(Path(relative_path).parts) > 1 else "unknown"
        counts[category] = counts.get(category, 0) + 1

    lines = ["**Documents by category:**", ""]
    for category in sorted(counts):
        lines.append(f"- **{category.title()}**: {counts[category]} files")
    lines.append(f"\n**Total**: {sum(counts.values())} files")
    return "\n".join(lines)


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
    return f"Total tokens for {model}: {len(token_count):,}"
