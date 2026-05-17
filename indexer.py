"""
indexer.py - Entry point do indexador.

Uso (conforme enunciado):
    python3 indexer.py -m <MEMORY> -c <CORPUS> -i <INDEX>

Argumentos:
    -m <MEMORY>: memoria disponivel em MB
    -c <CORPUS>: caminho do JSONL com o corpus
    -i <INDEX>: diretorio onde salvar os indices

Output:
    JSON no stdout com:
        - Index Size (MB)
        - Elapsed Time (s)
        - Number of Lists
        - Average List Size

Logs de progresso vao para STDERR para nao poluir o JSON final.
"""

import argparse
import json
import os
import sys
import time

from src.config.indexer import BLOCKS_DIR_NAME, INVERTED_INDEX_FILENAME
from src.indexing.merger import cleanup_blocks, merge_blocks
from src.indexing.spimi import SPIMIOrchestrator
from src.indexing.writer import compute_statistics, write_doc_index, write_lexicon
from src.preprocessing.nltk_setup import ensure_nltk_data
from src.utils.memory import MemoryMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web search engine indexer (SPIMI + external k-way merge).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        type=int,
        required=True,
        metavar="MEMORY",
        help="Available memory in megabytes",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=True,
        metavar="CORPUS",
        help="Path to the corpus JSONL file",
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        metavar="INDEX",
        help="Path to the output index directory",
    )
    # Flag opcional para desenvolvimento; nao faz parte do enunciado.
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="(dev) limita numero de docs lidos (para testes rapidos)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if args.m <= 0:
        print(f"erro: memoria invalida: {args.m}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.c):
        print(f"erro: corpus nao encontrado: {args.c}", file=sys.stderr)
        sys.exit(1)
    # Cria diretorio do indice se nao existir.
    os.makedirs(args.i, exist_ok=True)


def main():
    args = parse_args()
    validate_args(args)

    # NLTK precisa estar pronto antes das threads.
    ensure_nltk_data()

    # Cronometra desde aqui (depois do setup, que pode incluir download).
    start_time = time.perf_counter()

    # === Etapa 1: SPIMI ===
    blocks_dir = os.path.join(args.i, BLOCKS_DIR_NAME)
    memory = MemoryMonitor(budget_mb=args.m)

    print(
        f"[indexer] iniciando SPIMI: corpus={args.c}, mem={args.m}MB",
        file=sys.stderr,
    )
    spimi = SPIMIOrchestrator(
        corpus_path=args.c,
        blocks_dir=blocks_dir,
        memory=memory,
        max_docs=args.max_docs,
    )
    block_paths, doc_index = spimi.run()
    print(
        f"[indexer] SPIMI concluido: {len(block_paths)} blocks, "
        f"{len(doc_index)} docs",
        file=sys.stderr,
    )

    # === Etapa 2: k-way merge externo ===
    inverted_path = os.path.join(args.i, INVERTED_INDEX_FILENAME)
    print(f"[indexer] iniciando merge externo de {len(block_paths)} blocks",
          file=sys.stderr)
    lexicon = merge_blocks(block_paths, inverted_path)
    print(
        f"[indexer] merge concluido: {len(lexicon)} termos no lexicon",
        file=sys.stderr,
    )

    # === Etapa 3: cleanup dos blocks parciais ===
    cleanup_blocks(block_paths)
    # Remove o diretorio blocks/ se ficou vazio
    try:
        if os.path.isdir(blocks_dir) and not os.listdir(blocks_dir):
            os.rmdir(blocks_dir)
    except OSError:
        pass

    # === Etapa 4: persiste lexicon e document index ===
    write_lexicon(lexicon, args.i)
    write_doc_index(doc_index, args.i)
    print(f"[indexer] indice persistido em {args.i}", file=sys.stderr)

    # === Etapa 5: estatisticas ===
    elapsed = time.perf_counter() - start_time
    stats = compute_statistics(args.i, lexicon)
    stats["Elapsed Time"] = round(elapsed, 2)
    # Reordena chaves para casar com o exemplo do enunciado.
    output = {
        "Index Size": stats["Index Size"],
        "Elapsed Time": stats["Elapsed Time"],
        "Number of Lists": stats["Number of Lists"],
        "Average List Size": stats["Average List Size"],
    }

    # JSON final no stdout (unica coisa que vai pro stdout!)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
