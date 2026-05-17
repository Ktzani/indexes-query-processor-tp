"""
Constantes do processador de queries.
"""

# Numero maximo de resultados retornados por query.
TOP_K = 10

# Hiperparametros classicos do BM25 (Robertson et al.).
# k1 controla saturacao do TF (valores 1.2 a 2.0 sao tipicos).
# b controla normalizacao por tamanho do documento (0.75 eh classico).
BM25_K1 = 1.2
BM25_B = 0.75

# Encoding usado em arquivos de queries.
TEXT_ENCODING = "utf-8"
