import tensorflow as tf
import yaml
import argparse
import utils as mf
from losses import mae_cor, correlate
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import product
import time
from pathlib import Path
import pandas as pd
import datetime
import sys
plt.style.use('config/genome_research.mplstyle')

def prediction(model, sequences, winsize, output_idx=2):
    if len(sequences) == 0:
        raise ValueError("sequences is empty")

    batch = np.vstack(
        mf.sliding_window_view(
            sequences, (winsize, 4), axis=(1, 2)
        ).squeeze(2)
    )
    return model.predict_on_batch(batch)[:, output_idx]

def get_background(N_bg, seq_path, seq_length, lower_bound, upper_bound, winsize, predict_size):

    seq = mf.loadnp(seq_path)
    if upper_bound is not None:
        assert upper_bound <=  seq.shape[0] - seq_length, f"upper_bound is set too high, and will overflow. max_value : {seq.shape[0] - seq_length}"

    lower_bound =  0 if lower_bound is None else lower_bound
    upper_bound = seq.shape[0] - seq_length if upper_bound is None else upper_bound

    assert lower_bound >= 0, "lower_bound must be non-negative!"
    assert lower_bound <= upper_bound, "lower_bound have to be smaller than upper_bound!"

    background_coord = mf.space_random_opt(
        N_bg,
        predict_size + winsize - winsize % 2,
        lower_bound = lower_bound,
        upper_bound = upper_bound
    )
    ranges = mf.create_ranges(background_coord, background_coord + seq_length)
    return seq[ranges].reshape(len(background_coord), seq_length, 4), background_coord

def predict_kmers(model,
                  kmers,
                  background_sequences,
                  reps,
                  winsize,
                  predict_size,
                  tot_length,
                  output_idx
                  ):
    
    t0 = time.time()
    cpt = 0
    results = []
    for r in reps:
        results.append([])
        insert_len = kmers.shape[1] * r
        center = winsize // 2 + predict_size // 2
        start = center - insert_len // 2
        end = start + insert_len

        motif_block = np.tile(kmers, (1, r, 1))
        motif_block = motif_block.reshape(len(kmers), insert_len, 4)
        motif_block = motif_block[None, ...]
        motif_block = np.tile(motif_block, (background_sequences.shape[0], 1, 1, 1))

        mutated = np.tile(background_sequences[:, None, :, :], (1, len(kmers), 1, 1))
        mutated[:, :, start:end, :] = motif_block
        mutated = mutated.transpose(1, 0, 2, 3)

        for m_idx, m in enumerate(mutated):
            res_tmp = []
            for idx_batch in range(0, len(m), 4):
                seqs = m[idx_batch: idx_batch + 4]
                res = prediction(model, seqs, winsize, output_idx)
                res = res.reshape(-1, predict_size + 1)
                res_tmp.append(res)
                cpt += len(res)
                mf.loadbar(cpt, tot_length, t0=t0)

            results[-1].append(np.vstack(res_tmp))
    results = np.array(results)
    return results

def generate_plots(save_file, plot_dir, reps):
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    kmers_file = np.load(save_file)
    kmersynth = {k:kmers_file[k] for k in kmers_file.files}
    gc_content = np.mean(np.sum(kmersynth['kmers'][:, :, [1, 2]], axis=2), axis=1)

    score = np.mean(
        np.abs(
            kmersynth['predictions'] - kmersynth['bg_predictions'][None, ...]
            ),
            axis=-1
        )

    preds_over_bg = np.mean(kmersynth["predictions"], axis=2)
    bg_mean = np.mean(kmersynth["bg_predictions"], axis=0)

    kmer_names = ["".join(mf.BASES[np.argmax(x, axis=1)]) for x in kmersynth["kmers"]]

    n_rep, n_kmer, n_bg = score.shape

    df = pd.DataFrame({
        "nb_rep": np.repeat(reps, n_kmer * n_bg),
        "kmer": np.tile(np.repeat(kmer_names, n_bg), n_rep),
        "bg": np.tile([f"bg_{i}" for i in range(n_bg)], n_rep * n_kmer),
        "score": score.reshape(-1),
        "gc_content" : np.tile(np.repeat(gc_content, n_bg), n_rep)
    })
    df.to_csv(plot_dir / "summary.csv", index=False)

    # VIOLIN PLOT
    fig, ax = plt.subplots(figsize=(2, 2))

    # Groupes
    labels = sorted(df["gc_content"].unique())
    groups = [
        df[df["gc_content"] == g]["score"].values
        for g in labels
    ]


    parts = ax.violinplot(groups, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#87CEEB")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    parts["cmedians"].set_color("red")

    # Axes
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45)
    ax.tick_params(fontsize=10)
    
    ax.set_title("Score distribution by GC content")
    ax.set_xlabel("GC content", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)

    plt.tight_layout()
    mf.savefig(
        str(plot_dir / "violinplot"),
        fig
    )
    plt.close(fig)

    metaplot_path = (plot_dir / "kmers_metaplots")
    metaplot_path.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(reps)))
    print()
    print("Generating plots...")
    print()
    t0 = time.time()
    for i in range(preds_over_bg.shape[1]):
        fig = plt.figure()
        kmer_name = "".join(mf.BASES[np.argmax(kmersynth["kmers"][i], axis=1)])
        plt.title(kmer_name)
        plt.ylim(0, 1)
        plt.plot(bg_mean, color="k", linestyle=":", linewidth=0.5, label="background")
        for j, pred in enumerate(preds_over_bg[:, i]):
            plt.plot(preds_over_bg[j, i], label = reps[j], alpha=0.8, color=colors[j])
        # plt.legend()

        mf.savefig(
            str(metaplot_path / kmer_name),
            fig
        )
        plt.close(fig)
        mf.loadbar(i, preds_over_bg.shape[1], t0=t0)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=False, type=str, help="Path to YAML config file"
    )
    parser.add_argument(
        "--plot_only", required=False, type=str, help="Path to an existing run directory (or .npz file) to regenerate plots from"
    )
    args = parser.parse_args()

    if args.plot_only:
        run_dir = Path(args.plot_only)
        if run_dir.is_file() and run_dir.suffix == ".npz":
            save_file = run_dir
            run_dir = run_dir.parent
        else:
            save_file = run_dir / "synthetic_kmers.npz"

        config_path = run_dir / "kmersynth_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in {run_dir}. It's needed for reps parameter.")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        reps = cfg.get("repetitions_number")
        
        plot_dir = run_dir / "kmers_plots"
        generate_plots(save_file, plot_dir, reps)
        print("Done regenerating plots.")
        sys.exit(0)

    if not args.config:
        parser.error("--config is required unless --plot_only is used")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    #SEED
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # MODEL
    model_path = Path(cfg.get("model_path"))
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"mae_cor": mae_cor, "correlate": correlate},
    )
    output_idx = int(cfg.get("output_idx", 0))
    winsize = model.input_shape[1]

    #PARAMETERS
    reps = cfg.get("repetitions_number")
    monomer_size = int(cfg.get("monomer_size", 2))
    predict_size = int(cfg.get("predict_size", 1000))

    #BACKGROUND
    N_bg = int(cfg.get("N_bg", 1))
    seq_path = cfg.get("seq_path")
    lower_bound = cfg.get("lower_bound")
    upper_bound = cfg.get("upper_bound")

    kmers = np.array(list(product(np.eye(4), repeat=monomer_size)))
    tot_length = len(reps) * len(kmers) * N_bg
    seq_length = winsize + predict_size

    #SYTEM
    run_dir = Path(cfg.get("run_dir") or model_path.parent)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_dir / f"synthetic_kmers_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "kmersynth_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    make_plots = bool(cfg.get("make_plot", False))

    #COMPUTATION
    background_sequences, background_coords = get_background(N_bg, seq_path, seq_length, lower_bound, upper_bound, winsize, predict_size)
    background_predictions = np.vstack(
        [prediction(model, bg_s[None,:], winsize, output_idx) for bg_s in background_sequences]
        )
    
    results = predict_kmers(model,
                  kmers,
                  background_sequences,
                  reps,
                  winsize,
                  predict_size,
                  tot_length,
                  output_idx
                  )
    save_file = run_dir / "synthetic_kmers.npz"
    np.savez_compressed(save_file, kmers=kmers, predictions = results, bg_predictions = background_predictions, bg_coords = background_coords)

    if make_plots:
        plot_dir = (run_dir / "kmers_plots")
        generate_plots(save_file, plot_dir, reps)
