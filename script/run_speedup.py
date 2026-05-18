"""
Experimento de speedup variando o numero de threads do indexer.

Para cada N em THREAD_COUNTS:
    - Roda o indexer.py via subprocess com --threads N
    - Captura stdout (JSON) e stderr (logs)
    - Extrai metricas: tempo, num blocks, postings/bloco
    - Salva tudo em data/experiments/

Output:
    data/experiments/
        threads_<N>.log         # log completo (stderr)
        threads_<N>.json        # JSON final do indexer
        summary.csv             # tabela consolidada
        summary.json            # mesmas metricas em JSON

Uso:
    python experiments/run_speedup.py

    # Para mudar parametros:
    python experiments/run_speedup.py --max-docs 1000000 --threads 4 8 16

Tempo estimado:
    ~5-7 min por nivel de threads.
    Para [4, 8, 16, 32] = ~25-30 min total.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Configuracao padrao
DEFAULT_THREADS = [4, 8, 16, 32]
DEFAULT_MAX_DOCS = 500_000
DEFAULT_MEMORY = 1024
DEFAULT_CORPUS = "data/corpus/entities.jsonl"

# Pastas (relativas a raiz do projeto)
EXP_DIR = Path("data/experiments")
INDEX_BASE = "data/indexes/experiment_t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimento de speedup do indexer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=DEFAULT_THREADS,
        help="Lista de niveis de threads a testar",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help="Numero maximo de docs por run",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=DEFAULT_MEMORY,
        help="Budget de memoria em MB",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS,
        help="Caminho do corpus JSONL",
    )
    parser.add_argument(
        "--keep-indexes",
        action="store_true",
        help="Mantem os indices gerados (default: apaga para economizar disco)",
    )
    return parser.parse_args()


def run_indexer(
    threads: int,
    max_docs: int,
    memory: int,
    corpus: str,
    index_dir: str,
) -> tuple[dict, str, float]:
    """
    Roda o indexer e captura output.

    Retorna:
        (json_stats, full_log, wall_time)
    """
    cmd = [
        sys.executable, "indexer.py",
        "-m", str(memory),
        "-c", corpus,
        "-i", index_dir,
        "--max-docs", str(max_docs),
        "--threads", str(threads),
    ]
    print(f"  [run] {' '.join(cmd)}", flush=True)

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    wall_time = time.perf_counter() - start

    if proc.returncode != 0:
        print(f"  [erro] returncode={proc.returncode}", flush=True)
        print(f"  [stderr] {proc.stderr[-500:]}", flush=True)
        return {}, proc.stderr, wall_time

    # Parse JSON da ultima linha do stdout
    stdout_lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
    if not stdout_lines:
        return {}, proc.stderr, wall_time

    last_line = stdout_lines[-1]
    try:
        stats = json.loads(last_line)
    except json.JSONDecodeError:
        print(f"  [aviso] nao conseguiu parsear JSON: {last_line!r}", flush=True)
        return {}, proc.stderr, wall_time

    return stats, proc.stderr, wall_time


def count_blocks_in_log(log_text: str) -> int:
    """Conta linhas 'flush -> block_' no log para saber quantos blocks foram gerados."""
    return len(re.findall(r"flush -> block_", log_text))


def main():
    args = parse_args()

    # Confere que o corpus existe
    if not os.path.isfile(args.corpus):
        print(f"erro: corpus nao encontrado: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    # Cria diretorio de experiments
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experimento de Speedup - SPIMI Indexer")
    print(f"Corpus:    {args.corpus}")
    print(f"Max docs:  {args.max_docs}")
    print(f"Memory:    {args.memory} MB")
    print(f"Threads:   {args.threads}")
    print(f"Keep idx:  {args.keep_indexes}")
    print("=" * 60)
    print()

    results = []

    for n in args.threads:
        print(f"=== Run com {n} thread(s) ===", flush=True)

        index_dir = f"{INDEX_BASE}{n}"
        log_path = EXP_DIR / f"threads_{n}.log"
        json_path = EXP_DIR / f"threads_{n}.json"

        # Limpa indice anterior se existir
        if os.path.isdir(index_dir):
            shutil.rmtree(index_dir)

        stats, stderr_log, wall_time = run_indexer(
            threads=n,
            max_docs=args.max_docs,
            memory=args.memory,
            corpus=args.corpus,
            index_dir=index_dir,
        )

        # Salva log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stderr_log)

        if stats:
            # Salva JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

            num_blocks = count_blocks_in_log(stderr_log)

            row = {
                "Threads": n,
                "ElapsedTime": stats.get("Elapsed Time", 0),
                "WallTime": round(wall_time, 2),
                "IndexSize": stats.get("Index Size", 0),
                "NumberOfLists": stats.get("Number of Lists", 0),
                "AverageListSize": stats.get("Average List Size", 0),
                "Blocks": num_blocks,
            }
            results.append(row)

            print(f"  Tempo (interno): {row['ElapsedTime']}s")
            print(f"  Tempo (wall):    {row['WallTime']}s")
            print(f"  Index Size:      {row['IndexSize']} MB")
            print(f"  Blocks:          {row['Blocks']}")
            print()
        else:
            print(f"  [erro] sem JSON valido para {n} threads", flush=True)

        # Limpa indice (salvo se --keep-indexes)
        if not args.keep_indexes and os.path.isdir(index_dir):
            shutil.rmtree(index_dir)

    # Salva summary.csv
    csv_path = EXP_DIR / "summary.csv"
    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        # Salva summary.json tambem
        json_summary_path = EXP_DIR / "summary.json"
        with open(json_summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Calcula speedup relativo ao primeiro run
        baseline = results[0]["ElapsedTime"]

        print("=" * 60)
        print("RESUMO")
        print("=" * 60)
        print(f"{'Threads':>8} {'Time(s)':>10} {'Speedup':>10} {'Eff%':>8} {'Blocks':>8}")
        print("-" * 60)
        for row in results:
            speedup = baseline / row["ElapsedTime"] if row["ElapsedTime"] > 0 else 0
            efficiency = (speedup / row["Threads"]) * 100 if row["Threads"] > 0 else 0
            print(
                f"{row['Threads']:>8} "
                f"{row['ElapsedTime']:>10.1f} "
                f"{speedup:>9.2f}x "
                f"{efficiency:>7.1f}% "
                f"{row['Blocks']:>8}"
            )
        print()
        print(f"Resultados salvos em:")
        print(f"  {csv_path}")
        print(f"  {json_summary_path}")
        print(f"  {EXP_DIR}/threads_*.log")
        print(f"  {EXP_DIR}/threads_*.json")
    else:
        print("[erro] nenhum run completou com sucesso", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()