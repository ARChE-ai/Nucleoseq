import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logomaker as lm
import utils as mf
import pyBigWig as pbw
from time import time
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
import os
from pathlib import Path

plt.rcParams.update({
    # Font — picks the first one it finds installed
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],

    # Force every text element to the same size
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "figure.titlesize": 10,

    # Keep text as real, editable text in the SVG — not outlines
    "svg.fonttype": "none",
})

# --- Paths: resolved relative to the repo root, matching the PATHS/REPO_ROOT
#     convention used in notebooks/figures_new_clean.ipynb. This script lives at
#     REPO_ROOT/src/fimo_by_motif_plots.py, so the repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"

# Same run-directory names as PATHS["xstreme_dir"] / PATHS["mnase_run_dir"] /
# PATHS["chem_run_dir"] in figures_new_clean.ipynb — keep these two in sync if
# you rename a run directory.
XSTREME_DIR = DEMO_DIR / "test" / "xstreme250326_default_maxw30_JASPAR2026"
MNASE_RUN_DIR = DEMO_DIR / "test" / "MNase_17chroms_CORRECTED_1.5prctCovered_globalWeightsOptim_20260303_142239"
CHEM_RUN_DIR = DEMO_DIR / "test" / "Chemical_17chroms_CORRECTED_global_weights_20260304_205232"

COMBINED_MEME_PATH = XSTREME_DIR / "combined.meme"
mnase_ism_path = MNASE_RUN_DIR / "_ism_outputs" / "ism.bw"
chem_ism_path = CHEM_RUN_DIR / "_ism_outputs" / "ism.bw"
FIMO_MOTIS_PATH = XSTREME_DIR / "fimo_by_motif"

# --- Load Canonical PWMs from combined.meme ---
def load_all_meme_pwms(meme_path):
    pwms = {}
    current_motif = None
    matrix_rows = []

    with open(meme_path, 'r') as f:
        for line in f:
            if line.startswith('MOTIF'):
                if current_motif is not None and len(matrix_rows) > 0:
                    pwms[current_motif] = np.array(matrix_rows)
                parts = line.strip().split()
                current_motif = parts[2] if len(parts) >= 3 else parts[1]
                matrix_rows = []
            elif current_motif is not None:
                parts = line.strip().split()
                if not parts or line.startswith('letter-prob') or line.startswith('URL'):
                    continue
                try:
                    row = [float(x) for x in parts]
                    if len(row) == 4:
                        matrix_rows.append(row)
                except ValueError:
                    pass
    if current_motif is not None and len(matrix_rows) > 0:
        pwms[current_motif] = np.array(matrix_rows)
    return pwms

# Global parsing
CANONICAL_PWMS = load_all_meme_pwms(COMBINED_MEME_PATH)

def compute_individual_corrs(canonical_entropy, hit_isms):
    N = hit_isms.shape[0]
    corrs = np.zeros(N)
    for i in range(N):
        mask = ~np.isnan(canonical_entropy) & ~np.isnan(hit_isms[i])
        if np.sum(mask) > 3:
            rho, _ = stats.spearmanr(canonical_entropy[mask], hit_isms[i][mask])
            corrs[i] = rho
        else:
            corrs[i] = np.nan
    return corrs

def make_logo_df(pwms, **lm_kwargs):
    if len(pwms.shape)>2:
        pwms = np.mean(pwms, axis=0)

    assert len(pwms.shape)==2
    assert pwms.shape[1]==4
    return pd.DataFrame(pwms, columns=mf.BASES[:4])

def make_auto_bins(x, bins=100):
    return np.linspace(np.min(x), np.max(x), bins)

one_hot_dict = dict(zip("ACGT", np.eye(4)))
def one_hot(seq):
    return np.vstack([one_hot_dict.get(s, np.zeros(4)) for s in seq])


def make_plots(df_logo, entropy, ranges, ism_bw_path, save_format, filtered_df, canonical_entropy):
    ism_bw = pbw.open(str(ism_bw_path))
    ism = mf.zscore(ism_bw.values("chr19", 0, -1, numpy=True))

    hit_isms = ism[ranges]
    meta_ism = np.mean(hit_isms, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(4, 1.5))
    lm.Logo(entropy*df_logo, ax=axs[0])

    entropy = entropy.ravel()
    a = axs[0].twinx()
    a.plot(meta_ism, color="k")
    a.set_yticks([])

    axs[1].scatter(entropy, meta_ism)
    mask = ~np.isnan(entropy) & ~np.isnan(meta_ism)

    axs[1].scatter(entropy, meta_ism, alpha=0.5, s=15, color='k', label='Data points')

    rho, p_val = stats.spearmanr(entropy[mask], meta_ism[mask])
    stats_text = f"$\\rho = {rho:.3f}$, $p = {p_val:.2e}$"
    axs[1].text(0.05, 1.1, stats_text, transform=axs[1].transAxes,
                verticalalignment='top')

    axs[1].set_xlabel("Seq. Info.")
    axs[1].set_ylabel("ISM Score")
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position('right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    motif_name = filtered_df.motif_alt_id.iloc[0]
    title = f"{motif_name}, N={len(filtered_df)}"
    plt.suptitle(title, y=1.10, fontsize=10)
    plt.savefig(save_format.format(motif_name) + ".svg", bbox_inches="tight")
    plt.savefig(save_format.format(motif_name)+ ".pdf", dpi = 300, bbox_inches="tight")
    ism_bw.close()
    plt.close(fig)

    # Calculate individual correlations using canonical entropy
    individual_rhos = compute_individual_corrs(canonical_entropy, hit_isms)
    return rho, p_val, individual_rhos

def launch_plot(fimo_number):
    try:
        df = pd.read_csv(f"{FIMO_MOTIS_PATH}/{fimo_number}/fimo.tsv", sep="\t")
    except Exception as e:
        # Silently fail if motif has no matches to keep terminal output clean
        return [fimo_number, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    df = df[df.sequence_name == "chr19"]
    if len(df) == 0:
        return [fimo_number, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    len_motif = len(df.matched_sequence.iloc[0])
    df["norm_score"] = df.score/len_motif
    motif_name = df.motif_alt_id.iloc[0]

    filter_df = (df.norm_score >= 0.5)
    filtered_df = df[filter_df].copy()

    if len(filtered_df) == 0:
        return [fimo_number, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    filtered_df["pwm"] = filtered_df["matched_sequence"].apply(one_hot)
    pwms = np.stack(filtered_df["pwm"].values)
    df_logo = make_logo_df(pwms+1e-6)
    entropy_observed = 2 - mf.H(df_logo.values)[:, None]
    strand_p = filtered_df.strand=="-"
    ranges=mf.create_ranges(filtered_df.start.values, filtered_df.stop.values+1).reshape((-1, len_motif))
    ranges[strand_p] = ranges[strand_p, ::-1]

    # Get canonical entropy
    canonical_pwm = CANONICAL_PWMS.get(motif_name)
    if canonical_pwm is not None:
        canonical_entropy = (2 - mf.H(canonical_pwm)).ravel()
    else:
        # Fallback to observed entropy if canonical not found
        canonical_entropy = entropy_observed.ravel()

    try:
        mn_rho, mn_pval, mn_indiv_rhos = make_plots(
            df_logo,
            entropy_observed,
            ranges,
            mnase_ism_path,
            f"{FIMO_MOTIS_PATH}/_plots/{fimo_number}_{{}}_mnase",
            filtered_df,
            canonical_entropy
        )

        ch_rho, ch_pval, ch_indiv_rhos = make_plots(
            df_logo,
            entropy_observed,
            ranges,
            chem_ism_path,
            f"{FIMO_MOTIS_PATH}/_plots/{fimo_number}_{{}}_chem",
            filtered_df,
            canonical_entropy
        )

        # Save independent correlations CSV per motif
        filtered_df["mnase_rho"] = mn_indiv_rhos
        filtered_df["chem_rho"] = ch_indiv_rhos
        out_csv_dir = f"{FIMO_MOTIS_PATH}/_plots/csv"
        os.makedirs(out_csv_dir, exist_ok=True)
        filtered_df[["sequence_name", "start", "stop", "strand", "score", "norm_score", "mnase_rho", "chem_rho", "motif_alt_id"]].to_csv(
            os.path.join(out_csv_dir, f"{fimo_number}_{motif_name}_single_hits.csv"), index=False
        )

        return [fimo_number, motif_name, len(filtered_df), mn_rho, mn_pval, ch_rho, ch_pval]
    except Exception as e:
        print(f"Error with motif {fimo_number}: {e}")
        return [fimo_number, motif_name, len(filtered_df), np.nan, np.nan, np.nan, np.nan]

if __name__ == "__main__":
    motif_range = range(1, 629)

    results = Parallel(n_jobs=12)(
        delayed(launch_plot)(fimo_number)
        for fimo_number in tqdm(motif_range, desc="Processing motifs", unit="motif")
    )

    output_df = pd.DataFrame(results, columns=["fimo_number", "motif_name", "N", "mn_rho", "mn_pval", "ch_rho", "ch_pval"])
    output_df.to_csv(f"{FIMO_MOTIS_PATH}/_plots/_fimo_correlations.csv", index=False)
