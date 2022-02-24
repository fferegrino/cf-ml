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

def binarize_target(upstream, product):
    test = pd.read_csv(upstream["split"]["test"])
    val = pd.read_csv(upstream["split"]["val"])
    train = pd.read_csv(upstream["split"]["train"])


    train_y = np.where(train['speaker'] == "amlo", 1, 0)
    val_y = np.where(val['speaker'] == "amlo", 1, 0)
    test_y = np.where(test['speaker'] == "amlo", 1, 0)

    np.savetxt(product['train_y'], train_y.astype(int), fmt='%i')
    np.savetxt(product['val_y'], val_y.astype(int), fmt='%i')
    np.savetxt(product['test_y'], test_y.astype(int), fmt='%i')
