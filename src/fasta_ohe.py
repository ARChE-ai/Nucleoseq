# fasta_multi_to_onehot_npy.py
# FASTA multi-contigs -> one-hot (L,4) uint8 saved as .npy (memmap-friendly)
# - Parses mm10-like headers: "... chromosome 10, ..." -> chr10
# - Handles X/Y/M/MT
# - Fallback: first token after ">" sanitized
# - Optionally keep only canonical chromosomes (chr1-19, chrX, chrY, chrM) for mouse

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Set

import numpy as np

BASE_TO_COL = {
    ord("A"): 0,
    ord("C"): 1,
    ord("G"): 2,
    ord("T"): 3,
}

# Detect "... chromosome 10 ..." patterns in headers like your mm10.fa
CHR_RE = re.compile(rb"\bchromosome\s+([0-9]+|x|y|m|mt)\b", re.IGNORECASE)


def parse_contig_name(header_line: bytes) -> str:
    """
    header_line includes the leading '>'.
    Returns a filesystem-safe contig name.
    Priority:
      1) If header contains 'chromosome <id>' -> chr<id> (with MT -> M)
      2) Else first whitespace-delimited token after '>' (sanitized)
    """
    m = CHR_RE.search(header_line)
    if m:
        chrom = m.group(1).decode("ascii", errors="ignore").upper()
        if chrom == "MT":
            chrom = "M"
        return f"chr{chrom}"

    token = header_line[1:].strip().split()[0].decode("ascii", errors="ignore")
    # Sanitize minimal: avoid awkward characters in filenames
    token = token.replace("|", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    return token.lower()


def write_onehot_chr(seq_bytes: bytes, out_path: Path) -> None:
    """
    seq_bytes: ASCII bytes, uppercase, without newlines.
    Writes (L,4) uint8 one-hot to out_path (.npy).
    Non-ACGT -> all zeros.
    """
    L = len(seq_bytes)
    arr = np.zeros((L, 4), dtype=np.uint8)

    b = np.frombuffer(seq_bytes, dtype=np.uint8)
    for base_ord, col in BASE_TO_COL.items():
        arr[b == base_ord, col] = 1

    np.save(out_path, arr)


def fasta_to_onehot_npy_per_contig(
    fasta_path: str,
    out_dir: str,
    keep_contigs: Optional[Set[str]] = None,
) -> None:
    """
    Convert a multi-contig FASTA into one .npy per contig (memmap-friendly).

    keep_contigs:
      - None: keep everything
      - set of contig names to keep (e.g., {'chr1','chr2',...})
    """
    fasta_path = Path(fasta_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contig_name: Optional[str] = None
    chunks: List[bytes] = []

    def flush() -> None:
        nonlocal contig_name, chunks
        if contig_name is None:
            return

        if keep_contigs is not None and contig_name not in keep_contigs:
            # Skip writing this contig
            chunks = []
            return

        seq_bytes = b"".join(chunks)
        out_path = out_dir / f"{contig_name}.npy"
        print(f"[write] {contig_name} length={len(seq_bytes):,} -> {out_path}")
        write_onehot_chr(seq_bytes, out_path)
        chunks = []

    with open(fasta_path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                flush()
                contig_name = parse_contig_name(line)
            else:
                s = line.strip().upper()
                if s:
                    chunks.append(s)

    flush()


def mouse_mm10_canonical_contigs() -> Set[str]:
    # mm10 typically: chr1..chr19, chrX, chrY, chrM
    return {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY", "chrM"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Path to a multi-contig FASTA")
    parser.add_argument("--out", required=True, help="Output directory for .npy contigs")
    parser.add_argument(
        "--mm10_canonical_only",
        action="store_true",
        help="If set, only writes chr1-19, chrX, chrY, chrM (skips random/unplaced contigs)",
    )
    args = parser.parse_args()

    keep = mouse_mm10_canonical_contigs() if args.mm10_canonical_only else None
    fasta_to_onehot_npy_per_contig(args.fasta, args.out, keep_contigs=keep)
