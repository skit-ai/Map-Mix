from pyparsing import Dict
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

metrics = {} # since batch probabilties might get calculated out of order

class DataMapsCallback(pl.callbacks.Callback):
    """
    A pytorch lightning callback, implementing `Data Maps` described in the publication (https://arxiv.org/abs/2009.10795).
    """

    def __init__(self):
        # print("Init..")
        self.confidence = None
        self.variance = None
        self.correctness = None


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # print("On batch end..")
        # print(torch.cuda.current_device())
        # print(f"Batch idx: {batch_idx}")
        # print(f"dataloader idx: {dataloader_idx}")
        #print(batch)
        batch_probs = outputs['probs']   # batch_probs = instance, classes (2 x 14)
        batch_labels = outputs['labels'] # (2 x 1)
        filenames = outputs['filenames']
        # print(batch_labels)

        for ix, filename in enumerate(filenames):
            true_label_probs = batch_probs[ix][batch_labels[ix]]
            corrects = np.argmax(batch_probs[ix]) == batch_labels[ix]
            if filename in metrics:
                metrics[filename]['probs'] =  np.concatenate([metrics[filename]['probs'], true_label_probs.reshape(-1, 1)], axis=0)
                metrics[filename]['corrects'] =  np.concatenate([metrics[filename]['corrects'], corrects.reshape(-1, 1)], axis=0)
            else:
                metrics[filename] = {'probs': true_label_probs.reshape(-1, 1), 'corrects': corrects.reshape(-1, 1)}


    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        """
        Store the mean and std of probabilities across epochs for each instance after training ends.
        Also store correctness = mean of correct/total labels across epochs.
        """
        # Store the training artefacts


def embed_datamaps_into_dataframe(df, metrics):
    """
    Store training artefacts in a dataframe
    """

    for key in metrics.keys():
            metrics[key]['confidence'] = np.mean(metrics[key]['probs'])
            metrics[key]['variability'] = np.std(metrics[key]['probs'])
            metrics[key]['correctness'] = np.mean(metrics[key]['corrects'])

    # print(len(metrics))
    # print(len(df))

    assert len(metrics) == df.shape[0]

    df['probs'] = ""
    df['probs'] = df['probs'].astype('object')

    for i in range(len(df)):
        filename = df.loc[i, 'audiopath']
        df.loc[i, "confidence"] = metrics[filename]['confidence']
        df.loc[i, "variability"] = metrics[filename]['variability']
        df.loc[i, "correctness"] = metrics[filename]['correctness']
        df.loc[i, 'probs'] = metrics[filename]['probs'].tolist()

        # Euclidean distance between origin and (variability, confidence). Lower label scores correspond to hard-to-learn regions.
        df.loc[i, 'label_score'] = (df.loc[i, "variability"] ** 2 + df.loc[i, "confidence"] ** 2) ** 0.5

    return df

def generate_maps_plots_from_dataframe(df: pd.DataFrame, dir_: str):
    unique_labels = df['class'].unique().tolist()
    #print(unique_labels)
    fig, axs = plt.subplots(len(unique_labels), 1, figsize=(6, 6 * len(unique_labels)))

    box_style = {"boxstyle": "round", "facecolor": "white", "ec": "black"}
    for ix, label in enumerate(unique_labels):
        axs[ix].scatter(
            x=df.loc[df['class'] == label]["variability"].to_numpy(),
            y=df.loc[df['class'] == label]["confidence"].to_numpy(),
            marker="x",
            s=30,
            c=df.loc[df['class'] == label]["correctness"].to_numpy(),
        )
        axs[ix].set_title(f"datamap for label = {label}")
        axs[ix].set_xlabel("variability")
        axs[ix].set_ylabel("confidence")
        axs[ix].text(
            0.14,
            0.84,
            "easy-to-learn",
            transform=axs[ix].transAxes,
            verticalalignment="top",
            bbox=box_style, 
        )
        axs[ix].text(
            0.75,
            0.5,
            "ambiguous",
            transform=axs[ix].transAxes,
            verticalalignment="top",
            bbox=box_style,
        )
        axs[ix].text(
            0.14,
            0.14,
            "hard-to-learn",
            transform=axs[ix].transAxes,
            verticalalignment="top",
            bbox=box_style,
        )
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    df.to_csv(os.path.join(dir_, "datamaps-metrics-1.csv"), index=False)
    plt.savefig(os.path.join(dir_, "datamaps-1.png"), format="png")