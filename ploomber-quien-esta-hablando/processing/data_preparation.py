from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def split(upstream, product):
    dialogs = pd.read_csv(upstream["extract-dialogs"]["dialogs"])
    rest, test = train_test_split(dialogs, test_size=0.2, stratify=dialogs["speaker"])
    train, val = train_test_split(rest, test_size=0.2, stratify=rest["speaker"])

    test.to_csv(product["test"], index=False)
    train.to_csv(product["train"], index=False)
    val.to_csv(product["val"], index=False)

def binarise_speaker(dataframe):
    return np.where(dataframe["speaker"] == "amlo", 1, 0)

def binarise_target(upstream, product):
    val = pd.read_csv(upstream["split"]["val"])
    train = pd.read_csv(upstream["split"]["train"])

    train_y = binarise_speaker(train)
    val_y = binarise_speaker(val)

    np.savetxt(product['train_y'], train_y.astype(int), fmt='%i')
    np.savetxt(product['val_y'], val_y.astype(int), fmt='%i')
