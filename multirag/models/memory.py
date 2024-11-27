from collections import deque
from typing import Any, Tuple

class ContextualMemory:
    def __init__(self, max_size: int = 5):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, query: Any, results: Any):
        self.buffer.append((query, results))
        
    def get_recent_contexts(self, k: int = None) -> List[Tuple[Any, Any]]:
        k = k or len(self.buffer)
        return list(self.buffer)[-k:]
