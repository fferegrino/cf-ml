import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from text_processing import tokenize
import logging
import pickle

logger = logging.getLogger("training")


def load_data(file):
    dialogs = pd.read_csv(file, index_col=0)
    dialogs["speaker"] = np.where(dialogs["speaker"] == "amlo", "politico", "medico")
    return dialogs


def split_dataset(dialogs):
    rest, test = train_test_split(dialogs, test_size=0.2, stratify=dialogs["speaker"])
    train, val = train_test_split(rest, test_size=0.2, stratify=rest["speaker"])
    return train, val, test


@click.command()
@click.argument("dataset_path", type=click.Path())
@click.argument("max_features", type=click.INT)
@click.argument("binary", type=click.BOOL)
@click.argument("max_iter", type=click.INT)
@click.argument("class_weight")
@click.option("--verbose", "-v", is_flag=True)
def train(dataset_path, max_features, binary, max_iter, class_weight, verbose):
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("loaded dataset")
    dataset = load_data(dataset_path)

    logger.debug("splitting data")
    train, val, test = split_dataset(dataset)

    dialogs_train = train["dialog"]
    dialogs_val = val["dialog"]
    dialogs_test = test["dialog"]

    train_y = np.where(train["speaker"] == "politico", 1, 0)
    val_y = np.where(val["speaker"] == "politico", 1, 0)
    test_y = np.where(test["speaker"] == "politico", 1, 0)

    vectorizador_real = CountVectorizer(binary=binary, analyzer=tokenize, max_features=max_features)

    logger.debug("fitting vectorizer")
    vectorizador_real.fit(dialogs_train)

    train_x = vectorizador_real.transform(dialogs_train)
    val_x = vectorizador_real.transform(dialogs_val)
    test_x = vectorizador_real.transform(dialogs_test)

    lr = LogisticRegression(max_iter=max_iter, class_weight=class_weight)

    logger.debug("fitting logistic regression")
    lr.fit(train_x, train_y)

    train_pred = lr.predict(train_x)
    val_pred = lr.predict(val_x)

    training_accuracy = accuracy_score(train_y, train_pred)
    validation_accuracy = accuracy_score(val_y, val_pred)

    print(f"Training accuracy:   {training_accuracy:0.2%}")
    print(f"Validation accuracy: {validation_accuracy:0.2%}")

    # Accuracy on the holdout set
    test_pred = lr.predict(test_x)
    test_accuracy = accuracy_score(test_y, test_pred)
    print(f"Test accuracy:   {test_accuracy:0.2%}")

    with open("vectorizer.pkl", "wb") as wb:
        pickle.dump(vectorizador_real, wb)

    with open("model.pkl", "wb") as wb:
        pickle.dump(lr, wb)


if __name__ == "__main__":
    train()
