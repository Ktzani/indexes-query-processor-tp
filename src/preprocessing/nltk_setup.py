"""
Garante que os dados necessarios do NLTK (punkt para tokenizacao,
stopwords para filtro) estejam disponiveis.

Estrategia: tenta usar; se falhar com LookupError, baixa apenas o
necessario. Idempotente: chamadas subsequentes nao re-baixam.

IMPORTANTE: NLTK busca dados em ~/nltk_data (Linux/Mac) ou
%APPDATA%/nltk_data (Windows). O download eh feito uma vez por
ambiente, nao por execucao.
"""

import sys
import nltk


def _ensure_resource(resource_path: str, resource_name: str):
    """
    Garante que um recurso NLTK esta disponivel. Tenta localizar; se
    nao encontrar, baixa silenciosamente.

    Parametros:
        resource_path: caminho NLTK do recurso (e.g., 'tokenizers/punkt')
        resource_name: nome do pacote a baixar (e.g., 'punkt')
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"[nltk_setup] baixando {resource_name}...", file=sys.stderr)
        nltk.download(resource_name, quiet=True)


def ensure_nltk_data():
    """
    Garante que punkt (tokenizer) e stopwords (lista de stopwords)
    estao disponiveis. Idempotente.
    """
    _ensure_resource("tokenizers/punkt", "punkt")
    # NLTK 3.9+ separou alguns dados; punkt_tab eh necessario tambem.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("[nltk_setup] baixando punkt_tab...", file=sys.stderr)
        nltk.download("punkt_tab", quiet=True)
    _ensure_resource("corpora/stopwords", "stopwords")
