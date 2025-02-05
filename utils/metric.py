"""
With consideration to the contest on Kaggle
"""

import torch
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import log_loss

CFG = {
    "image_weight": 0.07361963,
    "exam_weights": [
        0.0736196319,
        0.2346625767,
        0.0782208589,
        0.06257668712,
        0.1042944785,
        0.06257668712,
        0.1042944785,
        0.1877300613,
        0.09202453988,
    ],
}


def bce(predictions, truth):
    return -(truth * np.log(predictions) + (1.0 - truth) * np.log(1.0 - predictions))


def rsna_metrics(y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes):
    label_weights = np.array(CFG["exam_weights"]).reshape((1, -1))
    img_weights = CFG["image_weight"]

    total_loss = 0.0
    total_weights = 0.0

    for y_img, y_exam, pred_img, pred_exam, size in zip(
            y_true_img, y_true_exam, y_pred_img, y_pred_exam, sizes
    ):
        y_img = y_img[:len(pred_img)]

        exam_loss = bce(pred_exam, y_exam)
        exam_loss = np.sum(exam_loss * label_weights, 1)[0]

        qi = np.sum(y_img)
        image_loss = bce(pred_img,y_img).mean()
        image_loss = np.sum(img_weights*qi*image_loss)

        total_loss += exam_loss + image_loss
        total_weights += label_weights.sum() + img_weights*qi

    final_weights = total_loss/total_weights
    return final_weights


class ConfusionMatrixDisplay:
    """
    Adapted from sklearn :
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    """

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, cmap="viridis", figsize=(10, 10), normalize=None, ax=None, fig=None):

        # Display colormap
        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)

        # Display values
        self.text_ = np.empty_like(cm, dtype=object)
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.3f}"
            self.text_[i, j] = ax.text(
                j, i, text, ha="center", va="center", color=color
            )

        # Display legend
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )

        ax.set_ylabel("True label", fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.tick_params(axis="both", which="minor", labelsize=11)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=40)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_cm(y_pred,y_true,normalize=None,display_labels=None,cmap="viridis",figsize=(10,10),ax=None,fig=None):
    cm = sklearn.metrics.confusion_matrix(y_true,y_pred,normalize=normalize)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=display_labels)
    return display.plot(cmap=cmap,figsize=figsize,normalize=normalize,ax=ax,fig=fig)