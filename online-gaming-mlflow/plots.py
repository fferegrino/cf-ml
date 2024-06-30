import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


def create_confusion_matrices(label_encoder, y_train, y_pred_train, y_test, y_pred_test):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    cf_train = confusion_matrix(y_train, y_pred_train)
    cf_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(
        cf_train,
        annot=True,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        fmt="",
        cmap="Reds",
        cbar=False,
        square=True,
        linewidths=1.1,
        yticklabels=label_encoder.classes_,
        xticklabels=label_encoder.classes_,
        ax=ax[0],
    )
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    ax[0].set_title("Train", fontsize=12, fontweight="bold", color="red")
    sns.heatmap(
        cf_test,
        annot=True,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1.1,
        yticklabels=label_encoder.classes_,
        xticklabels=label_encoder.classes_,
        ax=ax[1],
    )
    ax[1].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    ax[1].set_title("Test", fontsize=12, fontweight="bold", color="blue")
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold", color="black")
    fig.tight_layout()
    return fig


def create_learning_curves(model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training accuracy")
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    ax.plot(
        train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="Validation accuracy"
    )
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curves")
    ax.legend(loc="lower right")
    ax.grid()
    return fig


def plot_feature_importance(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(x="feature", y="importance", kind="bar", ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    return fig
