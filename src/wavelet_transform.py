# -*- coding: utf-8 -*-
"""
Genome-wide Continuous Wavelet Transform (CWT) periodicity track.

Computes a per-base "wavelet power" signal from a per-chromosome prediction/
mutasome track, using a complex Morlet wavelet tuned to the mono-/di-
nucleosomal period range (see Methods: "Continuous Wavelet Transform for
Periodicity Detection"). Output is written as a single genome-wide bigWig
track via utils.create_bw.

Usage:
    python src/wavelet_transform.py --config config/config_wavelets.yaml
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pywt
import yaml

import utils as mf


def compute_wavelet_track(prediction, wavelet, min_period, max_period, scale_step_divisor=10):
    """
    Return the base-pair-resolved wavelet power spectrum for one chromosome's
    prediction track, summed over scales spanning [min_period, max_period] bp.
    """
    max_frequency = 1 / max_period
    min_frequency = 1 / min_period

    min_scale = pywt.scale2frequency(wavelet, min_frequency)
    max_scale = pywt.scale2frequency(wavelet, max_frequency)
    scales = np.arange(
        min_scale,
        max_scale,
        scale_step_divisor * (max_scale - min_scale) / (max_period - min_period),
    )

    cwtmatr, _ = pywt.cwt(prediction, scales, wavelet)
    power = np.abs(cwtmatr) ** 2
    return power.sum(axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args_cli = parser.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    args = SimpleNamespace(**cfg)

    bw = mf.create_bw(args.output_file, chrom_sizes_path=args.chrom_sizes_path)

    for num_chr in args.chr:
        prediction = mf.loadnp(args.input_template.format(chrom=num_chr))

        wave_transform = compute_wavelet_track(
            prediction,
            wavelet=args.wavelet,
            min_period=args.min_period,
            max_period=args.max_period,
            scale_step_divisor=cfg.get("scale_step_divisor", 10),
        )

        bw.addEntries(f"chr{num_chr}", 0, values=wave_transform, span=1, step=1)
        print(f"chr{num_chr} done ({len(wave_transform)} bp)")

    bw.close()
