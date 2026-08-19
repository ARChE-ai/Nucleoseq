import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig as pbw
import logomaker as lm
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
import time
# Assuming utils is an external file available in the environment
import utils as mf

# =======================
# CONFIGURATION
# =======================
config = {
    "chrom": "chr19",
    "S_": 1500,
    "n_trials": 50,
    "half_size": 1500,
    "winsize": 2001,
    "batch_size": 2048,
    "cross_mut_bw": "/home/maxime/Nucleoseq/demo/test/crossmut.bw",
    "jaspar_tsv": "/home/maxime/data/sequences/mm10/TF/JASPAR2024/frigg.uio.no/JASPAR/JASPAR_TFBSs/2024/mm10/MA1930.2.tsv.gz",
    "one_hot_dir": "/home/maxime/data/sequences/mm10/one_hot/",
    "model_dir": "/home/maxime/Nucleoseq/demo/test/MNase_17chroms_CORRECTED_1.5prctCovered_globalWeightsOptim_20260303_142239/best.h5", # Change this to your actual model directory
    "run_dir": None # Defaults to model_dir/output if None
}

def make_logo(seqs, ax=None, **lm_kwargs):
    if ax is None:
        ax = plt.gca()
    
    # Fix: properly check the shape instead of undefined variable
    if len(seqs.shape) > 2:
        seqs = seqs.mean(axis=0)
    
    # If sequences are still > 2D or need further processing
    # but normally one mean(axis=0) reduces (batch, L, 4) to (L, 4).
    # Removing the redondant seqs = seqs.mean(axis=0) that was doing it twice.
    
    entropy = mf.H(seqs)[:, None]
    plot_logo_df = pd.DataFrame((2-entropy)*seqs, columns=["A", "C", "G", "T"])
    return lm.Logo(plot_logo_df, ax=ax, **lm_kwargs)
    
def make_prediction(model, synthetic_seq, winsize=2001, batch_size=2048, verbose=True):
    batches = mf.sliding_window_view(synthetic_seq, (1, winsize, 4)).squeeze((2,3))
    num_win_per_seq = batches.shape[1]
    batch_per_pred = batch_size // num_win_per_seq

    # Prédiction
    result = []
    t0 = time.time()
    for i in range(0, batches.shape[0], batch_per_pred):
        batch = np.vstack(batches[i:i+batch_per_pred])
        pred = model.predict_on_batch(batch)[:, 2].reshape(-1, num_win_per_seq)
        result.append(pred)
        if verbose:
            mf.loadbar(i, batches.shape[0], t0)
    return np.vstack(result)

def get_coord(df_path, chrom, columns, size=1500):
    df = pd.read_csv(
        df_path,
        sep="\t", header=None, names=columns
    )
    
    df = df[df.Chromosome==chrom]
    strands = df["Strand"].values

    coords = df[["Start", "End"]].values
    coords = coords[np.arange(len(coords)), (strands == "-").astype(int)]
    coords = coords[:, None] + np.arange(-size, size)
    coords[strands=="-"] = coords[strands=="-", ::-1]

    return coords, strands

def insert_motif(seq, motif, p0=None):
    n_seq = seq.copy()
    n_motif = motif.copy()
    if len(n_seq.shape) == 2:
        n_seq = n_seq[None, ...]
    if len(motif.shape) == 2:
        n_motif = n_motif[None, ...]

    batch_size, seq_len, _ = n_seq.shape
    motif_batch, motif_len, _ = n_motif.shape

    if p0 is None:
        p0 = seq_len // 2 - motif_len // 2

    if motif_batch == 1:
        # broadcast the motif to all sequences
        n_seq[:, p0 : p0 + motif_len, :] = n_motif[0]
    elif motif_batch == batch_size:
        n_seq[:, p0 : p0 + motif_len, :] = n_motif
    else:
        raise ValueError(f"Incompatible batch sizes: seq {batch_size}, motif {motif_batch}")

    return n_seq

def fetch_and_filter_seqs(config):
    chrom = config["chrom"]
    S_ = config["S_"]
    columns = ['Chromosome', 'Start', 'End', 'Name', 'Score1', 'Score2', 'Strand']
    
    cross_mut = pbw.open(config["cross_mut_bw"]).values(chrom, 0, -1, numpy=True)
    seq = mf.loadnp(os.path.join(config["one_hot_dir"], f"{chrom}.npz"))
    
    pre_coords, pre_strands = get_coord(
        config["jaspar_tsv"], 
        chrom, columns=columns, size=S_
    )
    
    footprint = np.log10(cross_mut[pre_coords[:, S_-15:S_+40]])
    motif_filter = (footprint[:, 15:25] >= 0).any(axis=1)
    
    pre_seqs = seq[pre_coords]
    pre_seqs[pre_strands == "-"] = pre_seqs[pre_strands == "-", :, ::-1]
    pre_seqs = pre_seqs[:, S_:S_+33]

    mut_preseqs = pre_seqs[motif_filter]
    nmut_preseqs = pre_seqs[~motif_filter]
    
    return mut_preseqs, nmut_preseqs, motif_filter

def make_random_seq(size=3000):
    nucleotide = np.random.choice([0, 1, 2, 3], size=3000, p = [0.29, 0.21, 0.21, 0.29] )
    z = np.zeros((size, 4), dtype = np.int8)
    z[np.arange(size), nucleotide] = 1
    return z

def generate_backgrounds(model, config, N):
    n_trials = config["n_trials"]
    half_size = config["half_size"]
    winsize = config["winsize"]
    batch_size = config["batch_size"]
    
    bgs = []
    mn_bgs = []
    
    for i in range(n_trials):
        seq_check = []
        std_check = []
        i_check = 0
        mn_bg = np.zeros(2 * half_size)

        # while np.any((mn_bg > 1) | (mn_bg < 0)):
        #     # Assuming make_random_seq is available via mf or somewhere else 
        #     # Needs to be defined or imported. Using a placeholder or assuming it exists.
        #     # If not, ensure it's imported.
        bg = np.stack([make_random_seq(size=2 * half_size)] * N)
        
        mn_bg = make_prediction(model, bg[None, 0], winsize, batch_size, verbose=False)[0]
            
            # std_check.append(np.std(mn_bg))
            # seq_check.append(bg)
            
            # if i_check > 20:
            #     less_std = np.argmin(std_check)
            #     bg = seq_check[less_std]
            #     mn_bg = make_prediction(model, bg[None, 0], winsize, batch_size)[0]
            #     break
            # i_check += 1
            
        mn_bgs.append(mn_bg.copy())
        bgs.append(bg[0].copy())
        
    return np.vstack(mn_bgs), np.stack(bgs)

def run_experiment_and_predict(model, config, bgs, mn_bg_array, mut_preseqs, nmut_preseqs, N):
    n_trials = config["n_trials"]
    half_size = config["half_size"]
    winsize = config["winsize"]
    batch_size = config["batch_size"]

    dfs = []
    result_mn_full_mut = []
    result_mn_full_nmut = []
    full_mut_seqs = None
    full_nmut_seqs = None

    t0 = time.time()
    for i in range(n_trials):
        current_bg = np.stack([bgs[i]] * N)
        mn_bg = mn_bg_array[i]
        
        # Insert synthetics
        synthetic_seq_full_mut = insert_motif(current_bg[:mut_preseqs.shape[0]], mut_preseqs, p0=half_size)
        synthetic_seq_full_nmut = insert_motif(current_bg[:nmut_preseqs.shape[0]], nmut_preseqs, p0=half_size)
        
        if i == n_trials - 1:
            full_mut_seqs = synthetic_seq_full_mut
            full_nmut_seqs = synthetic_seq_full_nmut

        # Predictions (silenced inner loadbars to avoid spam)
        mn_tmp_mut = make_prediction(model, synthetic_seq_full_mut, winsize, batch_size, verbose=False)
        result_mn_full_mut.append(mn_tmp_mut)
        
        mn_tmp_nmut = make_prediction(model, synthetic_seq_full_nmut, winsize, batch_size, verbose=False)
        result_mn_full_nmut.append(mn_tmp_nmut)

        # Amplitudes
        amp_mnase_full_mut = np.sum(np.abs(mn_bg - mn_tmp_mut), axis=1)
        amp_mnase_full_nmut = np.sum(np.abs(mn_bg - mn_tmp_nmut), axis=1)
        
        bgs_idx = [i] * N
        ism_label = (["+"] * amp_mnase_full_mut.shape[0]) + (["-"] * amp_mnase_full_nmut.shape[0])
        
        dfs.append(pd.DataFrame({
            "MNase-seq amp.": np.hstack([amp_mnase_full_mut, amp_mnase_full_nmut]),
            "Background": bgs_idx,
            "ISM": ism_label,
            "motif": np.arange(N)
        }))
        
        mf.loadbar(i, n_trials, t0)

    df_amp = pd.concat(dfs, ignore_index=True)
    result_mn_full_mut = np.vstack(result_mn_full_mut)
    result_mn_full_nmut = np.vstack(result_mn_full_nmut)
    
    return df_amp, result_mn_full_mut, result_mn_full_nmut, full_mut_seqs, full_nmut_seqs

def plot_final_results(mn_bg, result_mn_full_mut, result_mn_full_nmut, synthetic_seq_full_mut, synthetic_seq_full_nmut, motif_filter, config):
    out_dir = config["run_dir"]
    n_trials = config["n_trials"]
    
    fig, axs = plt.subplots(
        3, 2, sharex="col", figsize=(10, 10),
        gridspec_kw={"hspace": 0.05, "width_ratios": [1, 2]}
    )

    # Courbes moyennes
    axs[0][0].plot(np.mean(mn_bg, axis=0), label="Background", color="k", linestyle=":")
    axs[0][0].plot(np.mean(result_mn_full_mut, axis=0), label="Pred. ISM +", c="r")
    axs[0][0].plot(np.mean(result_mn_full_nmut, axis=0), alpha=0.7, label="Pred. ISM -", c="b")
    axs[0][0].set_ylim(0.1, 0.8)
    axs[0][0].set_title("MNase-seq model", fontsize=14, fontweight="bold")
    axs[0][0].legend(bbox_to_anchor=(1.05, 1.2), fontsize=14)

    # Heatmaps (lignes 1, 2) utilizing imshow instead of seaborn
    axs[1][0].imshow(result_mn_full_mut, cmap="Reds", aspect="auto", vmin=0.1, vmax=0.8, interpolation="none")
    axs[1][0].set_ylabel("ISM +", fontsize=12, fontweight="bold")

    axs[2][0].imshow(result_mn_full_nmut, cmap="Reds", aspect="auto", vmin=0.1, vmax=0.8, interpolation="none")
    axs[2][0].set_ylabel("ISM -", fontsize=12, fontweight="bold")

    # Logos
    axs[0][1].set_axis_off()
    make_logo(synthetic_seq_full_mut[:, 1500:1533], **{"ax": axs[1, 1]})
    make_logo(synthetic_seq_full_nmut[:, 1500:1533], **{"ax": axs[2, 1]})

    # Habillage
    axs[0][0].spines["top"].set_visible(False)
    axs[0][0].spines["right"].set_visible(False)
    axs[0][0].spines["bottom"].set_visible(False)
    axs[0][0].spines["left"].set_visible(False)
    
    # Axes heatmaps
    Nss = [0, sum(motif_filter), sum(~motif_filter)]
    for r in [1, 2]:
        Ns = Nss[r]
        quart = (Ns * n_trials) // 4
        offset_plot = Ns // 2

        axs[r][0].set_xticks([0, 500, 1000])
        axs[r][0].set_xticklabels([-500, 0, 500])
        axs[r][0].set_yticks([Ns//2, 2*quart+offset_plot, 4*quart-offset_plot])
        axs[r][0].set_yticklabels([f"Bg 0", f"Bg {n_trials//2}", f"Bg {n_trials-1}"])
        axs[r][0].set_rasterized(True)

    # Colonne logos : pas d'axes ni de cadres
    for r in [1, 2]:
        axs[r, 1].spines["top"].set_visible(False)
        axs[r, 1].spines["right"].set_visible(False)
        axs[r, 1].spines["left"].set_visible(False)
        axs[r, 1].set_yticks([])
    
    axs[1, 1].set_xticks([])
    axs[2, 1].set_xticks(range(0, 33, 5))
    
    axs[0, 0].set_xticks([])
    axs[1, 0].set_xticks([])
    
    plt.tight_layout()
    
    # Save the plot
    filepath = os.path.join(out_dir, "natural_onbg_ismPlusMinus")
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight')
    plt.savefig(f"{filepath}.svg", bbox_inches='tight', dpi=300)
    plt.close()

def plot_clustering(df_amp, config):
    out_dir = config["run_dir"]
    df_pivot = df_amp.pivot(index="Background", columns="motif", values="MNase-seq amp.")
    
    # Calculate distance and linkage for columns (motifs)
    dist_cols = ssd.pdist(df_pivot.T)
    Z_col = linkage(dist_cols, method='average')
    col_order = leaves_list(Z_col)
    
    # Calculate distance and linkage for rows (backgrounds)
    dist_rows = ssd.pdist(df_pivot)
    Z_row = linkage(dist_rows, method='average')
    row_order = leaves_list(Z_row)
    
    # Reorder data
    df_pivot_reordered = df_pivot.iloc[row_order, col_order]
    
    # Heatmap of the clustered data using matplotlib instead of seaborn
    fig, ax = plt.subplots(figsize=(8, 8))
    data_to_plot = np.log10(df_pivot_reordered.values)
    ax.imshow(data_to_plot, cmap="Reds", aspect="auto", interpolation="none")
    ax.set_title("Clustered Amplitudes (Log10)")
    ax.set_xlabel("Motif (ordered)")
    ax.set_ylabel("Background (ordered)")

    filepath_cluster = os.path.join(out_dir, "clustermap")
    plt.savefig(f"{filepath_cluster}.pdf", bbox_inches='tight')
    plt.savefig(f"{filepath_cluster}.svg", bbox_inches='tight', dpi=300)
    plt.close()

    # Create cluster map based on the columns linkage
    col_clusters = fcluster(Z_col, t=2, criterion="maxclust")
    cluster_map = pd.Series(col_clusters, index=df_pivot.columns, name="Cluster")

    df_amp_with_cluster = df_amp.merge(
        cluster_map.reset_index().rename(columns={"motif": "motif"}),
        on="motif"
    ).drop_duplicates(["motif"])

    # Crosstab heatmap using matplotlib instead of seaborn
    ct = pd.crosstab(df_amp_with_cluster.ISM.values, df_amp_with_cluster.Cluster.values, colnames=["Cluster"], rownames=["ISM"])
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(ct.values, cmap="Reds", aspect="auto", interpolation="none")
    
    # Add text annotations
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax2.text(j, i, str(ct.values[i, j]), ha="center", va="center", color="black" if ct.values[i, j] > np.max(ct.values)/2 else "red")
            
    ax2.set_xticks(np.arange(ct.shape[1]))
    ax2.set_yticks(np.arange(ct.shape[0]))
    ax2.set_xticklabels(ct.columns)
    ax2.set_yticklabels(ct.index)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("ISM")
    ax2.set_title("Cluster vs ISM Frequencies")

    filepath_heatmap = os.path.join(out_dir, "crosstab_heatmap")
    plt.savefig(f"{filepath_heatmap}.pdf", bbox_inches='tight')
    plt.savefig(f"{filepath_heatmap}.svg", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Setup directories
    if config["run_dir"] is None:
        # Si model_dir pointe vers un fichier (comme best.h5), prendre le dossier parent
        base_dir = config["model_dir"] if os.path.isdir(config["model_dir"]) else os.path.dirname(config["model_dir"])
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config["run_dir"] = os.path.join(base_dir, f"synthetic_CTCF_{timestamp}")
        
    os.makedirs(config["run_dir"], exist_ok=True)
    
    # Save the config
    config_path = os.path.join(config["run_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        print(f"Config saved to {config_path}")

    # Initialize / Load your model here
    import tensorflow as tf
    print(f"Loading model from {config['model_dir']}...")
    model = tf.keras.models.load_model(config["model_dir"], compile=False)

    print("Fetching and filtering sequences...")
    mut_preseqs, nmut_preseqs, motif_filter = fetch_and_filter_seqs(config)
    N = len(motif_filter)

    print("Generating backgrounds...")
    mn_bg, bgs = generate_backgrounds(model, config, N)

    print("Running experiments and predicting...")
    df_amp, res_mn_mut, res_mn_nmut, syn_mut, syn_nmut = run_experiment_and_predict(
        model, config, bgs, mn_bg, mut_preseqs, nmut_preseqs, N
    )

    print("Saving backgrounds and predictions...")
    np.savez_compressed(
        os.path.join(config["run_dir"], "predictions_and_backgrounds.npz"),
        background_seqs=bgs,
        background_preds=mn_bg,
        mutated_preds=res_mn_mut,
        non_mutated_preds=res_mn_nmut
    )

    df_amp.to_csv(os.path.join(config["run_dir"], "amplitudes.csv"), index=False)
    print("Amplitudes dataframe saved.")

    print("Plotting final results...")
    plot_final_results(mn_bg, res_mn_mut, res_mn_nmut, syn_mut, syn_nmut, motif_filter, config)

    print("Plotting clustering...")
    plot_clustering(df_amp, config)
    
    print("Done! All results saved in", config["run_dir"])

if __name__ == "__main__":
    main()
