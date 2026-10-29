# -*- coding: utf-8 -*-
"""
Custom losses/metrics for nucleosome models.
Usable in model.compile(...) and load_model(..., custom_objects=...).
"""
import tensorflow as tf
from tensorflow.keras import backend as K

def get_loss(name):
    if name=="mae_cor":
        return mae_cor
    elif name=="weighted_mae_cor":
        return weighted_mae_cor
    else:
        raise ValueError("Unknown loss")

def correlate(y_true, y_pred):
    """
		Calculate the correlation between the predictions and the labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = correlate)
		>>> load_model('file', custom_objects = {'correlate : correlate})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    return sigma_XY/(sigma_X*sigma_Y + K.epsilon())


def mae_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:
		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return ((1- cor) + mae)

# ---------------------------------------------------------------------------
# Per-bin weighted loss. Weights are packed with the labels in y_true
# ([labels | weights], shape (n, 2 * heads), see KDNAmulti_bw(pack_weights=True)),
# because Keras sample weights are applied after the loss: with a scalar loss
# such as mae_cor they only rescale each batch by mean(w).
# Zero-weight bins are excluded from both the MAE and the correlation.
# ---------------------------------------------------------------------------

def wmean(x, w):
    # epsilon: a batch with only zero weights gives a finite loss and gradients (no NaN)
    return K.sum(x * w) / (K.sum(w) + K.epsilon())


def _unpack(y_true, y_pred):
    n = K.int_shape(y_pred)[-1]
    return y_true[:, :n], y_true[:, n:], K.cast(y_pred, y_true.dtype)


def _wcor(y, y_pred, w):
    mx = wmean(y, w)
    my = wmean(y_pred, w)

    sx = wmean(K.square(y - mx), w)
    sy = wmean(K.square(y_pred - my), w)
    sxy = wmean((y - mx) * (y_pred - my), w)

    return sxy / (K.sqrt(sx * sy) + K.epsilon())


def weighted_correlate(y_true, y_pred):
    """
		Weighted Pearson correlation between predictions and labels (y_true = [labels | weights]).
	"""
    y, w, y_pred = _unpack(y_true, y_pred)
    return _wcor(y, y_pred, w)


def weighted_mae_cor(y_true, y_pred):
    """
		Weighted MAE + (1 - weighted Pearson correlation) (y_true = [labels | weights]).

		:Example:
		>>> model.compile(optimizer = 'adam', loss = weighted_mae_cor)
		>>> load_model('file', custom_objects = {'weighted_mae_cor': weighted_mae_cor})
	"""
    y, w, y_pred = _unpack(y_true, y_pred)
    return wmean(K.abs(y - y_pred), w) + 1 - _wcor(y, y_pred, w)
