import numpy as np
import os
import argparse
import tensorflow as tf
from math import log
import tqdm
from tensorflow.keras import metrics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def open_npz(path, key):
    data = np.load(path)[key]
    return data


def beamsLogScale(y, thresholdBelowMax):
    y_shape = y.shape  # shape is (#,256)

    for i in range(0, y_shape[0]):
        thisOutputs = y[i, :]
        logOut = 20 * np.log10(thisOutputs + 1e-30)
        minValue = np.amax(logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs[zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum(thisOutputs)
        y[i, :] = thisOutputs
    return y


def getBeamOutput(output_file):
    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file["output_classification"]

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0], num_classes)
    y = beamsLogScale(y, thresholdBelowMax)

    return y, num_classes


def custom_label(output_file, strategy="one_hot"):
    "This function generates the labels based on input strategies, one hot, reg"

    print("Reading beam outputs...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file["output_classification"]

    yMatrix = np.abs(yMatrix)

    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    y = yMatrix.reshape(yMatrix.shape[0], num_classes)
    y_non_on_hot = np.array(y)
    y_shape = y.shape

    if strategy == "one_hot":
        k = 1  # For one hot encoding we need the best one
        for i in range(0, y_shape[0]):
            thisOutputs = y[i, :]
            logOut = 20 * np.log10(thisOutputs)
            max_index = logOut.argsort()[-k:][::-1]
            y[i, :] = 0
            y[i, max_index] = 1

    elif strategy == "reg":
        for i in range(0, y_shape[0]):
            thisOutputs = y[i, :]
            logOut = 20 * np.log10(thisOutputs)
            y[i, :] = logOut
    else:
        print("Invalid strategy")
    return y_non_on_hot, y, num_classes


def over_k(true, pred):  # for TF2
    ####compute accuracy per K
    dicti = {}
    for kth in range(256):
        kth_accuracy = metrics.top_k_categorical_accuracy(true, pred, k=kth)
        with tf.compat.v1.Session() as sess:
            this = kth_accuracy.eval()
        dicti[kth] = sum(this) / len(this)
    return dicti


def througput_ratio(preds, y):
    ####compute throughput ratio
    throughputs = {}
    for k in tqdm(range(1, 256)):
        up = []
        down = []
        for exp in range(len(y)):
            true_1 = y[exp].argsort()[-1:][::-1]
            t1 = log(y[exp, true_1] + 1, 2)

            top_preds = preds[exp].argsort()[-k:][::-1]
            p1 = max([log(y[exp, t] + 1, 2) for t in top_preds])
            up.append(p1)
            down.append(t1)

        throughputs["choices_" + str(k)] = sum(up) / sum(down)
    return throughputs
