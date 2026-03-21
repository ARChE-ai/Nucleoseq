#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from Bio import SeqIO

import utils as mf


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.get("output_root") is None:
        output_root = Path(cfg["model_path"]).parent
    else:
        output_root = Path(cfg["output_root"])
    experiment_name = cfg.get("experiment_name", "synthetic_satellites")
    run_dir = output_root / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    for subdir in ["raw", "plots", "logs"]:
        (run_dir / subdir).mkdir(exist_ok=True)

    return run_dir


def copy_config(config_path: str, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def sanitize_name(name: str) -> str:
    keep = []
    for c in name:
        if c.isalnum() or c in ("-", "_", "."):
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def one_hot_encode_dna(seq: str) -> np.ndarray:
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    return np.array([mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq], dtype=np.int8)


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    valid = sum(base in "ACGT" for base in seq)
    if valid == 0:
        return np.nan
    gc = sum(base in "GC" for base in seq)
    return gc / valid


def smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    if window is None or window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def get_amplitude(profile: np.ndarray) -> float:
    return float(np.max(profile) - np.min(profile))

def shuffle_sequence(seq: str, k: int = 3) -> str:
    import ushuffle
    if k <= 1:
        arr = np.array(list(seq))
        np.random.shuffle(arr)
        return "".join(arr)
    
    seq_bytes = seq.encode('utf-8')
    shuffled_bytes = ushuffle.shuffle(seq_bytes, k)
    return shuffled_bytes.decode('utf-8')

def read_fasta(fasta_path: str) -> list:
    entries = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        entries.append({
            "header": record.description, 
            "sequence": str(record.seq).upper()
        })
    return entries


def repeat_to_length(seq: str, target_length: int) -> str:
    if len(seq) == 0:
        raise ValueError("Empty sequence cannot be repeated.")
    n_repeat = math.ceil(target_length / len(seq))
    return (seq * n_repeat)[:target_length]


def build_windows_from_sequence(seq: str, winsize: int, target_length: int) -> np.ndarray:
    repeated_seq = repeat_to_length(seq, target_length)
    one_hot = one_hot_encode_dna(repeated_seq)

    if one_hot.shape[0] < winsize:
        raise ValueError(
            f"Sequence length after repeat ({one_hot.shape[0]}) is smaller than winsize ({winsize})."
        )

    windows = mf.sliding_window_view(one_hot, (winsize, 4))
    windows = windows.squeeze(axis=1)
    return windows


def predict_profile(
    model,
    seq: str,
    winsize: int,
    target_length: int,
    output_idx: int,
    batch_size: int,
) -> np.ndarray:
    windows = build_windows_from_sequence(seq, winsize=winsize, target_length=target_length)
    preds = model.predict(windows, batch_size=batch_size, verbose=0)

    preds = np.asarray(preds)
    if preds.ndim == 1:
        profile = preds
    elif preds.ndim == 2:
        profile = preds[:, output_idx]
    else:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    return np.asarray(profile).ravel()

def load_model_from_config(cfg: dict):
    model_path = cfg["model_path"]
    return tf.keras.models.load_model(model_path, compile=False)


def compute_summary_stats(native_amp: float, perm_amps: np.ndarray) -> dict:
    perm_mean = float(np.mean(perm_amps))
    perm_std = float(np.std(perm_amps))
    perm_median = float(np.median(perm_amps))
    percentile = float(100.0 * np.mean(perm_amps <= native_amp))

    if perm_std > 0:
        zscore = float((native_amp - perm_mean) / perm_std)
    else:
        zscore = np.nan

    return {
        "native_amplitude": native_amp,
        "perm_mean_amplitude": perm_mean,
        "perm_std_amplitude": perm_std,
        "perm_median_amplitude": perm_median,
        "native_percentile": percentile,
        "native_zscore": zscore,
    }


def plot_histogram(
    native_amp: float,
    perm_amps: np.ndarray,
    seq_name: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(perm_amps, bins="auto", alpha=0.8)
    ax.axvline(native_amp, linewidth=2, color="r")
    ax.set_title(f"{seq_name} - native vs shuffled amplitudes")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    fig.tight_layout()
    mf.savefig(outpath, fig)
    plt.close(fig)


def plot_profile(
    native_profile: np.ndarray,
    seq_name: str,
    outpath: Path,
    perm_profiles=None,
    show_perm_envelope: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(native_profile))

    if show_perm_envelope and perm_profiles is not None and len(perm_profiles) > 0:
        p05 = np.percentile(perm_profiles, 5, axis=0)
        p50 = np.percentile(perm_profiles, 50, axis=0)
        p95 = np.percentile(perm_profiles, 95, axis=0)

        ax.fill_between(x, p05, p95, alpha=0.25, label="Shuffled 5-95%")
        ax.plot(x, p50, linewidth=1.5, alpha=0.9, label="Shuffled median", color="k")

    ax.plot(x, native_profile, linewidth=2, label="Native", color="r")
    ax.set_title(f"{seq_name} - prediction profile")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Prediction")
    ax.legend()
    fig.tight_layout()
    mf.savefig(outpath, fig)
    plt.close(fig)


def analyze_sequence(
    model,
    header: str,
    sequence: str,
    shuffled_sequences: list,
    cfg: dict,
) -> dict:
    winsize = model.input_shape[1]
    target_length = cfg["target_length"]
    output_idx = cfg["output_idx"]
    batch_size = cfg["batch_size"]
    n_perm = len(shuffled_sequences)
    smooth_window = cfg.get("smooth_window", 5)
    save_perm_profiles = cfg["save_perm_profiles"]

    native_profile = predict_profile(
        model=model,
        seq=sequence,
        winsize=winsize,
        target_length=target_length,
        output_idx=output_idx,
        batch_size=batch_size,
    )
    native_profile = smooth(native_profile, smooth_window)
    native_amp = get_amplitude(native_profile)

    perm_amps = []
    perm_profiles = [] if save_perm_profiles or cfg.get("show_perm_envelope", True) else None

    t0 = time.time()
    for cpt, shuffled_seq in enumerate(shuffled_sequences):
        shuffled_profile = predict_profile(
            model=model,
            seq=shuffled_seq,
            winsize=winsize,
            target_length=target_length,
            output_idx=output_idx,
            batch_size=batch_size,
        )
        shuffled_profile = smooth(shuffled_profile, smooth_window)

        perm_amps.append(get_amplitude(shuffled_profile))

        if perm_profiles is not None:
            perm_profiles.append(shuffled_profile)
        mf.loadbar(
            cpt, n_perm, t0, 
            "Analyzing sequence " + header
        )

    perm_amps = np.asarray(perm_amps, dtype=np.float32)
    perm_profiles = None if perm_profiles is None else np.asarray(perm_profiles, dtype=np.float32)

    summary = compute_summary_stats(native_amp, perm_amps)

    result = {
        "header": header,
        "sequence_length": len(sequence),
        "gc_fraction": gc_fraction(sequence),
        **summary,
        "native_profile": native_profile.astype(np.float32),
        "perm_amplitudes": perm_amps,
        "perm_profiles": perm_profiles,
    }
    return result


def save_sequence_results(result: dict, seq_dir: Path, cfg: dict) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)

    np.save(seq_dir / "native_profile.npy", result["native_profile"])
    np.save(seq_dir / "perm_amplitudes.npy", result["perm_amplitudes"])

    if cfg.get("save_perm_profiles", False) and result["perm_profiles"] is not None:
        np.save(seq_dir / "perm_profiles.npy", result["perm_profiles"])

    summary_keys = [
        "header",
        "sequence_length",
        "gc_fraction",
        "native_amplitude",
        "perm_mean_amplitude",
        "perm_std_amplitude",
        "perm_median_amplitude",
        "native_percentile",
        "native_zscore",
    ]
    summary_json = {k: result[k] for k in summary_keys}
    with open(seq_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare native satellite predictions to shuffled controls.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", None))

    run_dir = make_run_dir(cfg)
    copy_config(args.config, run_dir)

    model = load_model_from_config(cfg)
    fasta_entries = read_fasta(cfg["fasta_path"])
    shutil.copy2(cfg["fasta_path"], run_dir / "sequences.fa")
    if len(fasta_entries) == 0:
        raise ValueError(f"No FASTA entry found in {cfg['fasta_path']}")


    summary_rows = []

    for entry in fasta_entries:
        header = entry["header"]
        sequence = entry["sequence"]

        seq_name = sanitize_name(header)
        seq_dir = run_dir / "raw" / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        n_perm = cfg["n_perm"]
        kmer_conservation = cfg.get("kmer_conservation", 3)
        shuffled_sequences = [shuffle_sequence(sequence, k=kmer_conservation) for _ in range(n_perm)]

        with open(seq_dir / "shuffled_sequences.fa", "w") as f:
            for i, sseq in enumerate(shuffled_sequences):
                f.write(f">{header}_shuffled_{i}\n{sseq}\n")

        result = analyze_sequence(
            model=model,
            header=header,
            sequence=sequence,
            shuffled_sequences=shuffled_sequences,
            cfg=cfg,
        )

        save_sequence_results(result, seq_dir, cfg)

        plot_histogram(
            native_amp=result["native_amplitude"],
            perm_amps=result["perm_amplitudes"],
            seq_name=header,
            outpath=run_dir / "plots" / f"{seq_name}_hist",
        )

        plot_profile(
            native_profile=result["native_profile"],
            perm_profiles=result["perm_profiles"],
            seq_name=header,
            outpath=run_dir / "plots" / f"{seq_name}_profile",
            show_perm_envelope=cfg.get("show_perm_envelope", True),
        )

        summary_rows.append(
            {
                "header": result["header"],
                "sequence_length": result["sequence_length"],
                "gc_fraction": result["gc_fraction"],
                "native_amplitude": result["native_amplitude"],
                "perm_mean_amplitude": result["perm_mean_amplitude"],
                "perm_std_amplitude": result["perm_std_amplitude"],
                "perm_median_amplitude": result["perm_median_amplitude"],
                "native_percentile": result["native_percentile"],
                "native_zscore": result["native_zscore"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    print(f"Analysis finished. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()