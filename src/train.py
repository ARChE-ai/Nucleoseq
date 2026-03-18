# -*- coding: utf-8 -*-
"""
@author: maxime christophe
"""

import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import LossScaleOptimizer

import argparse
import datetime
import os

import utils as mf
import numpy as np
import pandas as pd
from generator_opt_determinist import KDNAmulti_bw
from losses import get_loss, correlate, mae_cor
from modeles import create_model

# import wandb
import shutil


day = datetime.datetime.now()

if __name__ == "__main__":
    tfimport_dict = mf.init_tf(tf)

    # ------------------------
    # CONFIG (YAML)
    # ------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # OUTPUT
    experiment_name = cfg.get("experiment_name", "experiment")
    output_dir = cfg.get("output_dir", "results")
    gpath = output_dir
    dirname = f"{experiment_name}_{day.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(gpath, dirname)
    os.makedirs(run_dir, exist_ok=True)

    print("Loaded config:", args.config)
    print("Run dir:", f"{gpath}/{dirname}")

    # # WANDB INIT
    # wandb.init(project=cfg.get("project", "nucleoseq"), name=f"{dirname}", config=cfg)

    # GENERAL
    random_seed = int(cfg.get("random_seed", 42))
    gpu_index = cfg.get("gpu_index", 0)
    workers = int(cfg.get("workers", -1))

    # DATA
    data_cfg = cfg["data"]
    seq = data_cfg["seq_path"]
    lab = data_cfg["label_path"]
    mask_train = data_cfg.get("mask_train", None)
    mask_val = data_cfg.get("mask_val", None)
    if mask_val is None:
        mask_val = mask_train

    # MODEL
    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "CNN_standard5H")
    winsize = int(model_cfg.get("winsize", 2001))
    batch_size = int(model_cfg.get("batch_size", 4096))
    epochs = int(model_cfg.get("epochs", 100))
    learning_rate = float(model_cfg.get("learning_rate", 1e-3))
    loss_name = model_cfg.get("loss", "mae_cor")
    loss = get_loss(loss_name)
    metrics_names = model_cfg.get("metrics", ["mae", "correlate"])
    steps = np.array(list(map(int, model_cfg.get("steps", [-500, -250, 0, 250, 500]))))

    # TRAINING
    train_cfg = cfg["training"]
    train_chr = list(map(int, train_cfg["train_chr"]))
    val_chr = list(map(int, train_cfg["val_chr"]))
    training_weights = train_cfg.get("apply_weights", "")
    train_dist = int(train_cfg.get("training_distance_between_windows", 300))
    val_dist = int(train_cfg.get("validation_distance_between_windows", 300))

    def scheduler(epoch, lr):
        if epoch < 2:
            return lr
        else:
            return lr * 0.5

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',     
        factor=0.2,            
        patience=2,         
        min_lr=1e-6,           
        verbose=1              
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if int(tfimport_dict["ver"].split(".")[1]) < 11:
        opt = LossScaleOptimizer(opt)
    metrics = [mae_cor, correlate]

    if len(tfimport_dict["devices"]) > 1:
        with tfimport_dict["strategy"].scope():
            model = create_model(
                model_name,
                optimizer=opt,
                input_shape=(winsize, 4),
                loss=loss,
                metrics=metrics,
            )
    else:
        model = create_model(
            model_name,
            optimizer=opt,
            input_shape=(winsize, 4),
            loss=loss,
            metrics=metrics,
        )

    print(model.summary())

    print("VAL GENERATOR")
    val_gen = KDNAmulti_bw(
        seq=seq,
        lab=lab,
        chr=val_chr,
        winsize=winsize,
        headsteps=steps,
        weights=False,
        batch_size=batch_size,
        mask=mask_val,
        rundir=f"{gpath}/{dirname}",
        workers=workers,
        training=False,
        dist_between_windows=val_dist,
    )

    print("TRAINING GENERATOR")
    train_gen = KDNAmulti_bw(
        seq=seq,
        lab=lab,
        chr=train_chr,
        winsize=winsize,
        headsteps=steps,
        weights=training_weights,
        batch_size=batch_size,
        mask=mask_train,
        rundir=f"{gpath}/{dirname}",
        workers=workers,
        training=True,
        dist_between_windows=train_dist,
    )

    mcp_save = ModelCheckpoint(
        f"{gpath}/{dirname}" + "/best.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    os.mkdir(os.path.join(run_dir, "epochs"))
    # every epoch
    ckpt_epoch = ModelCheckpoint(
        filepath=os.path.join(run_dir, "epochs", "epoch_{epoch:03d}.h5"),
        save_best_only=False,
        save_weights_only=True,  # mets True si tu veux juste les poids (fichiers plus légers)
        verbose=0,
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print(">>>>let's go")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=[mcp_save, ckpt_epoch, earlystop, reduce_lr],
        use_multiprocessing=True,
        workers=workers,
        max_queue_size=5,
    )

    model.save(f"{gpath}/{dirname}/model")
    hist_df = pd.DataFrame(history.history)
    with open(f"{gpath}/{dirname}/history", mode="w") as f:
        hist_df.to_csv(f)

    shutil.copy(args.config, os.path.join(run_dir, "train_config_used.yaml"))
