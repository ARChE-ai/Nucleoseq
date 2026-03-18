#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf

import utils as mf


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(cfg: dict) -> Path:
    output_root = Path(cfg["run"]["output_root"])
    experiment_name = cfg["run"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{experiment_name}"
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


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
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


def shuffle_sequence(seq: str, rng: np.random.Generator) -> str:
    arr = np.array(list(seq))
    rng.shuffle(arr)
    return "".join(arr.tolist())


def read_fasta(fasta_path: str) -> list[dict]:
    entries = []
    header = None
    seq_chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_chunks).upper()
                    entries.append({"header": header, "sequence": seq})
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        seq = "".join(seq_chunks).upper()
        entries.append({"header": header, "sequence": seq})

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
    model_path = cfg["model"]["model_path"]
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
    bins: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_amps, bins=bins, alpha=0.8)
    ax.axvline(native_amp, linewidth=2)
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
    perm_profiles: np.ndarray | None = None,
    show_perm_envelope: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(native_profile))

    if show_perm_envelope and perm_profiles is not None and len(perm_profiles) > 0:
        p05 = np.percentile(perm_profiles, 5, axis=0)
        p50 = np.percentile(perm_profiles, 50, axis=0)
        p95 = np.percentile(perm_profiles, 95, axis=0)

        ax.fill_between(x, p05, p95, alpha=0.25, label="Shuffled 5-95%")
        ax.plot(x, p50, linewidth=1.5, alpha=0.9, label="Shuffled median")

    ax.plot(x, native_profile, linewidth=2, label="Native")
    ax.set_title(f"{seq_name} - prediction profile")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def analyze_sequence(
    model,
    header: str,
    sequence: str,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    winsize = cfg["prediction"]["winsize"]
    target_length = cfg["prediction"]["target_length"]
    output_idx = cfg["prediction"]["output_idx"]
    batch_size = cfg["prediction"]["batch_size"]
    n_perm = cfg["shuffle"]["n_perm"]
    smooth_window = cfg["prediction"].get("smooth_window", 1)
    save_perm_profiles = cfg["save"].get("save_perm_profiles", False)

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
    perm_profiles = [] if save_perm_profiles or cfg["plots"].get("show_perm_envelope", True) else None

    for _ in range(n_perm):
        shuffled_seq = shuffle_sequence(sequence, rng)
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

    if cfg["save"].get("save_perm_profiles", False) and result["perm_profiles"] is not None:
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
    fasta_entries = read_fasta(cfg["input"]["fasta_path"])

    if len(fasta_entries) == 0:
        raise ValueError(f"No FASTA entry found in {cfg['input']['fasta_path']}")

    max_entries = cfg["input"].get("max_entries", None)
    if max_entries is not None:
        fasta_entries = fasta_entries[:max_entries]

    rng = np.random.default_rng(cfg.get("seed", None))

    summary_rows = []

    for entry in fasta_entries:
        header = entry["header"]
        sequence = entry["sequence"]

        seq_name = sanitize_name(header)
        seq_dir = run_dir / "raw" / seq_name

        result = analyze_sequence(
            model=model,
            header=header,
            sequence=sequence,
            cfg=cfg,
            rng=rng,
        )

        save_sequence_results(result, seq_dir, cfg)

        plot_histogram(
            native_amp=result["native_amplitude"],
            perm_amps=result["perm_amplitudes"],
            seq_name=header,
            outpath=run_dir / "plots" / f"{seq_name}_hist.png",
            bins=cfg["plots"].get("hist_bins", 40),
        )

        plot_profile(
            native_profile=result["native_profile"],
            perm_profiles=result["perm_profiles"],
            seq_name=header,
            outpath=run_dir / "plots" / f"{seq_name}_profile.png",
            show_perm_envelope=cfg["plots"].get("show_perm_envelope", True),
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