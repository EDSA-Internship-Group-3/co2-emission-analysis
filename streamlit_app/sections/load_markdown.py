from pathlib import Path


def load_markdown_file(file_path):
    with open(Path(file_path),'r',encoding='utf-8') as f:
        return f.read()
