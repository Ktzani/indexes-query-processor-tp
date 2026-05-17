"""
Representa uma posting (doc_id, tf) na inverted list.

Usamos NamedTuple por ser:
- Imutavel (evita bugs de mutacao acidental)
- Iteravel (pode desempacotar: doc_id, tf = posting)
- Indexavel (posting[0] = doc_id, posting[1] = tf)
- Lightweight (mesma performance de tupla pura)
"""

from typing import NamedTuple


class Posting(NamedTuple):
    """Uma posting em uma inverted list."""
    doc_id: int
    tf: int
