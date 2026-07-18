from pathlib import Path

import logging

from livekit.agents import function_tool

logger = logging.getLogger("file-tools")

# Define the root directory for file operations (the project root)
ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_workspace_path(relative_path: str) -> Path:
    return (ROOT_DIR / relative_path).resolve()


@function_tool(description="List files in a directory. Use '.' for the current directory.")
def list_files(directory: str = ".") -> str:
    """Lists files in the specified directory."""
    try:
        target_dir = _resolve_workspace_path(directory)
        if not str(target_dir).startswith(str(ROOT_DIR)):
            return "Error: Access denied. Directory is outside of workspace."

        if not target_dir.exists():
            return f"Error: Directory {directory} does not exist."

        files = [str(f.relative_to(ROOT_DIR)) for f in target_dir.iterdir()]
        return "\n".join(files) if files else "Directory is empty."
    except Exception as e:
        return f"Error listing files: {str(e)}"


@function_tool(description="Read the contents of a specific file.")
def read_file(file_path: str) -> str:
    """Reads and returns the content of a file."""
    try:
        target_file = _resolve_workspace_path(file_path)
        if not str(target_file).startswith(str(ROOT_DIR)):
            return "Error: Access denied. File is outside of workspace."

        if not target_file.is_file():
            return f"Error: {file_path} is not a file or does not exist."

        return target_file.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"


@function_tool(description="Overwrite a file with new content. Use carefully.")
def edit_file(file_path: str, content: str) -> str:
    """Writes content to a file, overwriting existing data."""
    try:
        target_file = _resolve_workspace_path(file_path)
        if not str(target_file).startswith(str(ROOT_DIR)):
            return "Error: Access denied. File is outside of workspace."

        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}."
    except Exception as e:
        return f"Error writing file: {str(e)}"


class FileManagementTools:
    """Backwards-compatible wrapper for file utilities."""

    def list_files(self, directory: str = ".") -> str:
        return list_files(directory)

    def read_file(self, file_path: str) -> str:
        return read_file(file_path)

    def edit_file(self, file_path: str, content: str) -> str:
        return edit_file(file_path, content)
