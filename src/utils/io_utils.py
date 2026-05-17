"""
Utilitarios de I/O. Encapsula leitura/escrita binaria do indice e
helpers de leitura de arquivos texto. Mantem encoding e endianness
consistentes em todo o projeto.

Endianness: usamos LITTLE-ENDIAN explicitamente para que o indice
gerado em uma plataforma seja portavel. Sem isso, o indice produzido
em Windows poderia nao ser legivel em outra plataforma de teste.

Tamanhos fixos:
    UINT16 = 2 bytes (term_length, ate 65535 chars; suficiente)
    UINT32 = 4 bytes (doc_id, tf, num_postings; suficiente para 4.6M docs)
"""

import struct

# Formatos struct (formato little-endian explicito com prefixo "<").
FMT_UINT16 = "<H"
FMT_UINT32 = "<I"
FMT_POSTING = "<II"  # (doc_id, tf), 8 bytes

SIZEOF_UINT16 = 2
SIZEOF_UINT32 = 4
SIZEOF_POSTING = 8


def write_uint16(f, value: int):
    """Escreve um uint16 little-endian."""
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"uint16 fora do range: {value}")
    f.write(struct.pack(FMT_UINT16, value))


def write_uint32(f, value: int):
    """Escreve um uint32 little-endian."""
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"uint32 fora do range: {value}")
    f.write(struct.pack(FMT_UINT32, value))


def read_uint16(f) -> int:
    """Le um uint16 little-endian. Retorna -1 se EOF."""
    data = f.read(SIZEOF_UINT16)
    if len(data) < SIZEOF_UINT16:
        return -1
    return struct.unpack(FMT_UINT16, data)[0]


def read_uint32(f) -> int:
    """Le um uint32 little-endian. Retorna -1 se EOF."""
    data = f.read(SIZEOF_UINT32)
    if len(data) < SIZEOF_UINT32:
        return -1
    return struct.unpack(FMT_UINT32, data)[0]


def write_posting(f, doc_id: int, tf: int):
    """Escreve uma posting (doc_id, tf) em 8 bytes."""
    f.write(struct.pack(FMT_POSTING, doc_id, tf))


def read_posting(f) -> tuple[int, int] | None:
    """Le uma posting de 8 bytes. Retorna None se EOF."""
    data = f.read(SIZEOF_POSTING)
    if len(data) < SIZEOF_POSTING:
        return None
    return struct.unpack(FMT_POSTING, data)


def file_size_mb(path: str) -> float:
    """Retorna o tamanho de um arquivo em MB."""
    import os
    return os.path.getsize(path) / (1024.0 * 1024.0)
