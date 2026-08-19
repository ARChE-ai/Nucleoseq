#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the genome-wide, integer-encoded sequence bigWig expected by
generator_opt_determinist.KDNAmulti_bw as `data.seq_path` in the training config
(see config/config_train.yaml and README.md Phase 0).

Each position's value is the base index: 0=A, 1=C, 2=G, 3=T, -1=N/unmapped
(matching KDNAmulti_bw.onehot_with_minus_one, which one-hot-encodes on the fly
and treats -1 as "no base").

Input: per-chromosome one-hot .npy arrays produced by fasta_ohe.py
       (data/sequences/chr{N}.npy, shape (L, 4), one-hot ACGT).
Output: a single genome-wide bigWig (default data/sequences/mm10.bw).

Usage
-----
    python src/fasta_ohe.py --fasta data/sequences/mm10.fa --out data/sequences --mm10_canonical_only
    python src/build_sequence_bw.py \
        --seq_dir data/sequences \
        --chrom_sizes data/sequences/mm10.chrom.sizes \
        --out data/sequences/mm10.bw
"""

import argparse
import numpy as np

from utils import create_bw


def build_sequence_bw(seq_dir, chrom_sizes, out_path, chromosomes):
    bw = create_bw(out_path, chrom_sizes)
    try:
        for chrom in chromosomes:
            npy_path = f"{seq_dir}/chr{chrom}.npy"
            print(f"[chr{chrom}] loading {npy_path}")
            onehot = np.load(npy_path)  # (L, 4), uint8, one-hot ACGT

            idx = np.argmax(onehot, axis=1).astype(np.float32)
            idx[onehot.sum(axis=1) == 0] = -1.0  # N / unmapped bases: no single base set

            print(f"[chr{chrom}] writing {len(idx):,} positions")
            bw.addEntries(f"chr{chrom}", 0, values=idx, span=1, step=1)
    finally:
        bw.close()
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_dir", required=True, help="Directory with chr{N}.npy one-hot arrays (from fasta_ohe.py)")
    parser.add_argument("--chrom_sizes", required=True, help="Path to a UCSC-style chrom.sizes file")
    parser.add_argument("--out", required=True, help="Output bigWig path (must not already exist)")
    parser.add_argument(
        "--chromosomes",
        default=",".join(str(i) for i in range(1, 20)),
        help="Comma-separated chromosome numbers to include, e.g. '1,2,3' (default: 1-19, autosomes only)",
    )
    args = parser.parse_args()

    chroms = [c.strip() for c in args.chromosomes.split(",") if c.strip()]
    build_sequence_bw(args.seq_dir, args.chrom_sizes, args.out, chroms)
