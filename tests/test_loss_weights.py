"""
Checks how per-bin weights reach the loss.

Run (CPU is enough):
    python tests/test_loss_weights.py      (or: pytest tests/test_loss_weights.py)
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import tensorflow as tf
from losses import mae_cor, weighted_mae_cor, weighted_correlate
from generator_opt_determinist import KDNAmulti_bw

rng = np.random.default_rng(0)
N, HEADS = 512, 5
y = rng.random((N, HEADS)).astype(np.float32)
y[rng.random((N, HEADS)) < 0.3] = 0  # zero-label bins
p = rng.random((N, HEADS)).astype(np.float32)
w = np.where(y == 0, 0, rng.uniform(0.2, 5, (N, HEADS))).astype(np.float32)  # like "auto"


def identity_model(loss, metrics=None):
    # prediction = input, so the loss is evaluated on (y, p) exactly
    model = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x, input_shape=(HEADS,))])
    model.compile(loss=loss, metrics=metrics)
    return model


def wloss(labels, weights, pred=p):
    return float(weighted_mae_cor(tf.constant(np.concatenate([labels, weights], axis=1)), tf.constant(pred)))


def test_mae_cor_sample_weight_only_rescales():
    # Documents the old behavior: mae_cor returns a scalar, so Keras sample weights
    # only multiply the batch loss by mean(w); they do not weight bins.
    model = identity_model(mae_cor)
    weighted = model.evaluate(p, y, sample_weight=w, batch_size=N, verbose=0)
    unweighted = float(mae_cor(tf.constant(y), tf.constant(p)))
    assert np.isclose(weighted, unweighted * w.mean(), rtol=1e-5)


def test_weighted_mae_cor_ignores_zero_weight_bins():
    y2 = y.copy()
    y2[w == 0] = rng.random(int((w == 0).sum()))
    assert np.isclose(wloss(y, w), wloss(y2, w), rtol=1e-6)


def test_weighted_mae_cor_unit_weights_equals_mae_cor():
    ones = np.ones_like(y)
    assert np.isclose(wloss(y, ones), float(mae_cor(tf.constant(y), tf.constant(p))), rtol=1e-5)


def test_weighted_mae_cor_binary_weights_equals_mae_cor_on_kept_bins():
    keep = (w > 0).astype(np.float32)
    subset = float(mae_cor(tf.constant(y[keep > 0]), tf.constant(p[keep > 0])))
    assert np.isclose(wloss(y, keep), subset, rtol=1e-5)


def test_weighted_mae_cor_all_zero_weights_is_finite():
    pred = tf.Variable(p)
    with tf.GradientTape() as tape:
        loss = weighted_mae_cor(tf.constant(np.concatenate([y, np.zeros_like(w)], axis=1)), pred)
    assert np.isfinite(float(loss))
    assert np.isfinite(tape.gradient(loss, pred).numpy()).all()


def test_packed_generator_output_trains():
    gen = KDNAmulti_bw.__new__(KDNAmulti_bw)  # only _output is needed, no bigWig files
    gen.pack_weights = True
    out = gen._output(p, y, w)
    assert len(out) == 2 and out[1].shape == (N, 2 * HEADS)

    gen.pack_weights = False
    assert len(gen._output(p, y, w)) == 3

    model = tf.keras.Sequential([tf.keras.layers.Dense(HEADS, activation="sigmoid", input_shape=(HEADS,))])
    model.compile(optimizer="adam", loss=weighted_mae_cor, metrics=[weighted_mae_cor, weighted_correlate])
    hist = model.fit(p, out[1], batch_size=128, epochs=1, verbose=0)
    assert np.isfinite(hist.history["loss"][0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
