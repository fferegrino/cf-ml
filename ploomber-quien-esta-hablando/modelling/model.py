from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import load_npz
import pickle

# %% tags=["parameters"]
upstream = ['binary-target', 'tokenised-text'] 
product = None

# %%
train_x = load_npz(upstream['tokenised-text']['train_x'])
val_x = load_npz(upstream['tokenised-text']['val_x'])

train_y = np.loadtxt(upstream['binary-target']['train_y'])
val_y = np.loadtxt(upstream['binary-target']['val_y'])

# %%
lr = LogisticRegression(max_iter=1000, class_weight="balanced")

lr.fit(train_x, train_y)

# %%

train_pred = lr.predict(train_x)
val_pred = lr.predict(val_x)

# %%
with open(product["model"], "wb") as wb:
    pickle.dump(lr, wb)