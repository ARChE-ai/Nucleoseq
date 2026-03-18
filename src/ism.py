# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:38 2022

@author: maxime christophe
"""

import argparse
import yaml
from types import SimpleNamespace
import os
import time
import numpy as np
import utils as mf
from utils import loadbar
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer


def mse_mutasome_opt(
    model,
    sequence,
    pos=None,
    size=None,
    winsize=2001,
    output_file: str = "mutasome",
    save: bool = True,
    result: bool = True,
    batch: int = 8192,
    reverse=False,
    write_batch=5_000_000,
    steps = [-500, -250, 0, 250, 500]
):
    # Faire varier le nombre de steps
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param positions: first position

    :param int size: length of desired mutasome
    :param str output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True
    :param int offset: locus of the first prediction
    """

    try:
        nb_heads = int(getattr(model, "output_shape", [None, 5])[-1])
        if not (1 <= nb_heads <= 1000):
            nb_heads = 5
    except Exception:
        nb_heads = 5
    seq = sequence

    if pos is None:
        pos = 0

    if size is None:
        size = len(seq) - pos - winsize

    t0 = time.time()

    mutation = np.tile(np.eye(4), (batch // 4, 1)).reshape((batch // 4, 4, 4))

    # mutate sequence
    MUTASOME_SCORE = []
    PREDICTION = []

    try:
        os.remove(output_file + "tmp")

    except FileNotFoundError:
        pass

    try:
        os.remove(output_file + "PREDICTIONtmp")
    except FileNotFoundError:
        pass

    with open(output_file + "tmp", "a") as mutscor_file:
        with open(output_file + "PREDICTIONtmp", "a") as pred_file:
            for cpt, p in enumerate(range(pos - winsize // 2, pos + size, batch // 4)):
                try:
                    seqs = mf.sliding_window_view(
                        seq[p : p + (batch // 4) + winsize - winsize % 2], (winsize, 4)
                    )  # .reshape((batch//4), winsize, 4)
                    repseq = np.repeat(seqs, 4, axis=1)
                    # np.expand_dims(seqs, axis=1)
                    repseq[:, :, winsize // 2, :] = mutation[: len(repseq)]
                    repseq = repseq.reshape((-1, winsize, 4))

                    # separate indices of lab and pred
                    idxlab = np.argmax(
                        seqs[:, :, winsize // 2], axis=-1
                    ).ravel() + np.arange(0, batch, 4)
                    idxpred = np.ones(batch, dtype=bool)
                    idxpred[idxlab] = 0

                    # Predict
                    rawpred = model(repseq, training=False).numpy()
                    res = rawpred[idxpred].reshape((batch // 4, 3, nb_heads))
                    lab_batch = rawpred[idxlab].reshape((batch // 4, 1, nb_heads))

                    # MSE
                    sumtmp = np.mean((res - lab_batch) ** 2, axis=2).ravel()
                    MUTASOME_SCORE.extend(list(sumtmp))
                    PREDICTION.extend(list(lab_batch[:, :, 2]))
                    loadbar(cpt, np.round(size / (batch // 4)), t0)

                    if cpt % write_batch == 0 and cpt > 0:
                        np.savetxt(
                            mutscor_file, np.array(MUTASOME_SCORE).reshape((-1, 3))
                        )
                        np.savetxt(pred_file, np.array(PREDICTION))
                        MUTASOME_SCORE = []
                        PREDICTION = []
                except ValueError:
                    break
                finally:
                    mutscor_file.flush()
                    pred_file.flush()

            np.savetxt(
                mutscor_file,
                np.array(MUTASOME_SCORE, dtype=np.float32).reshape((-1, 3)),
            )
            np.savetxt(pred_file, np.array(PREDICTION, dtype=np.float32))

    if save:
        MUTASOME_SCORE = np.zeros((len(seq), 3), dtype=np.float32)
        muttmp = np.loadtxt(output_file + "tmp")
        MUTASOME_SCORE[pos : pos + len(muttmp), :] = muttmp

        np.savez_compressed(output_file, np.array(MUTASOME_SCORE).reshape((-1, 3)))
        os.remove(output_file + "tmp")

        PREDICTION = np.zeros(len(seq))
        predtmp = np.loadtxt(output_file + "PREDICTIONtmp")
        PREDICTION[pos : pos + len(predtmp)] = predtmp
        np.savez_compressed(output_file + "PREDICTION", np.array(PREDICTION).ravel())

        os.remove(output_file + "PREDICTIONtmp")
        print("Saved - {}s".format(time.time() - t0))

    if result:
        return (
            np.array(PREDICTION, dtype=np.float16).ravel(),
            np.array(MUTASOME_SCORE, dtype=np.float16).reshape((-1, 3)),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config.yaml")
    args_cli = parser.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    args = SimpleNamespace(**cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        for gpu_instance in physical_devices:
            tf.config.experimental.set_memory_growth(gpu_instance, True)
    except Exception:
        pass

    model = tf.keras.models.load_model(
        os.path.join(f"{args.model}", "best.h5"),
        custom_objects={
            "mae_cor": None,
            "correlate": None,
            "LossScaleOptimizer": LossScaleOptimizer,
            "Adam": None,
        },
    )

    output_dir = os.path.join(args.model, "_ism_outputs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for chromosome in args.chr:
        seq = np.load(f"{args.seq_dir}/chr{chromosome}.npz")["arr_0"]
        mse_mutasome_opt(
            model=model,
            sequence=seq,
            output_file=f"{output_dir}/{chromosome}_{args.output_file}",
            save=True,
            result=False,
            batch=cfg.get("batch_size", 4096),
            pos=cfg.get("pos", 3000000),
            size=cfg.get("size", None),
            write_batch=cfg.get("write_batch", 500_000),
        )

