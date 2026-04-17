from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str  # filename or URL
    page: int | None
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
