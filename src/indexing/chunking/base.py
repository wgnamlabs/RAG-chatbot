from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Splits text into chunks.
        Args:
            text: The full text to split.
            metadata: Base metadata to attach to every chunk.
        Returns:
            List of Chunk objects.
        """
        pass
