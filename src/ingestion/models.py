from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str  # filename or URL
    page: int | None
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)
