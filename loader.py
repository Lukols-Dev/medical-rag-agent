import logging
from pathlib import Path
from typing import Iterator
import pypdf
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

logger = logging.getLogger(__name__)


class DocumentsLoader(BaseLoader):
    """
        Load Documents
    """

    def __init__(self, file_path: Path):
        """
        Initialize the document loader with configuration.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        Return single file
        """
        with pypdf.PdfReader(self.file_path)  as reader:
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                
                if not text.strip():
                    continue

                yield Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path.name,
                        "page": page_number
                    }
                )