#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build nprs.fasta / nprs.csv / crossmut.bw: candidate "cross-mutasome" peak
regions used as XSTREME input for motif discovery (see README.md Phase 3).

Combines the MNase and chemical-cleavage ISM tracks (geometric mean of their
z-scored, smoothed signal) into a single "cross-mutasome" track, finds local
peaks on it, and extracts a fixed-width window of sequence around each peak.

This is a fixed, script-ified version of demo/test/nprs.ipynb: the notebook
used `Path` and `height` without defining them and called `df.to_csv(...)`
without ever building `df` — fixed here. It also previously lived only under
demo/, which is entirely gitignored, so it wasn't tracked or reproducible by
anyone else; this script belongs in src/ instead.

Usage
-----
    python src/build_nprs_fasta.py \
        --mnase_ism_bw demo/test/<mnase_run>/_ism_outputs/ism.bw \
        --chem_ism_bw  demo/test/<chem_run>/_ism_outputs/ism.bw \
        --seq_bw data/sequences/mm10.bw \
        --chrom_sizes data/sequences/mm10.chrom.sizes \
        --out_crossmut_bw demo/test/crossmut.bw \
        --out_fasta demo/test/nprs.fasta \
        --out_csv demo/test/nprs.csv
"""

import argparse
import numpy as np
import pandas as pd
import pyBigWig as pbw
from scipy.signal import find_peaks

from utils import zscore, create_bw, BASES


def prep_chromosome(mut_bw, chrom):
    vals = mut_bw.values(chrom, 0, -1, numpy=True)
    zmask = vals != 0
    vals[zmask] = zscore(vals[zmask])
    vals = np.convolve(vals, np.ones(15) / 15, "same")
    vals = np.clip(vals, 0, a_max=None)
    return vals


def build_nprs(mnase_ism_bw, chem_ism_bw, seq_bw_path, chrom_sizes, out_crossmut_bw,
                out_fasta, out_csv, chromosomes, half_width, peak_height, peak_distance):
    mnase_mut = pbw.open(mnase_ism_bw)
    chem_mut = pbw.open(chem_ism_bw)
    seq_bw = pbw.open(seq_bw_path)
    crossmut_bw = create_bw(out_crossmut_bw, chrom_sizes)

    chromosomes_col, peak_col, start_col, end_col, height_col = [], [], [], [], []

    try:
        with open(out_fasta, "w") as write_file:
            for chrom in [f"chr{c}" for c in chromosomes]:
                print(f"[{chrom}] scoring")
                mnase_vals = prep_chromosome(mnase_mut, chrom)
                chem_vals = prep_chromosome(chem_mut, chrom)
                cross = np.sqrt(mnase_vals * chem_vals)
                crossmut_bw.addEntries(chrom, 0, values=cross, span=1, step=1)

                peaks, props = find_peaks(cross, height=peak_height, distance=peak_distance)
                print(f"[{chrom}] {len(peaks)} peaks")

                for i, p in enumerate(peaks):
                    lo, hi = p - half_width, p + half_width
                    write_file.write(f">{chrom}:{lo}-{hi}\n")
                    base_idx = seq_bw.values(chrom, lo, hi, numpy=True).astype(int)
                    write_file.write("".join(BASES[base_idx]) + "\n")

                    chromosomes_col.append(chrom)
                    peak_col.append(p)
                    start_col.append(lo)
                    end_col.append(hi)
                    height_col.append(props["peak_heights"][i])
    finally:
        crossmut_bw.close()
        mnase_mut.close()
        chem_mut.close()
        seq_bw.close()

    df = pd.DataFrame({
        "chrom": chromosomes_col,
        "peak": peak_col,
        "start": start_col,
        "end": end_col,
        "height": height_col,
    })
    df.to_csv(out_csv, index=False)
    print(f"Done -> {out_fasta}, {out_csv}, {out_crossmut_bw} ({len(df)} regions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnase_ism_bw", required=True)
    parser.add_argument("--chem_ism_bw", required=True)
    parser.add_argument("--seq_bw", required=True, help="Genome-wide sequence bigWig, see src/build_sequence_bw.py")
    parser.add_argument("--chrom_sizes", required=True)
    parser.add_argument("--out_crossmut_bw", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--chromosomes", default=",".join(str(i) for i in range(1, 20)))
    parser.add_argument("--half_width", type=int, default=15, help="Half-width of each extracted region (default 15 -> 30bp windows)")
    parser.add_argument("--peak_height", type=float, default=1.0)
    parser.add_argument("--peak_distance", type=int, default=20)
    args = parser.parse_args()

    chroms = [c.strip() for c in args.chromosomes.split(",") if c.strip()]
    build_nprs(
        args.mnase_ism_bw, args.chem_ism_bw, args.seq_bw, args.chrom_sizes,
        args.out_crossmut_bw, args.out_fasta, args.out_csv,
        chroms, args.half_width, args.peak_height, args.peak_distance,
    )
