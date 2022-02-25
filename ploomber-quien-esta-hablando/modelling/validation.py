from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# %% tags=["parameters"]
from processing.data_preparation import binarise_speaker

upstream = ['split', 'model', 'tokenised-text']
product = None

# %%
with open(upstream['tokenised-text']['vectoriser'], "rb") as rb:
    vectoriser = pickle.load(rb)

test_data = pd.read_csv(upstream['split']['test'])

test_y = binarise_speaker(test_data)
test_x = vectoriser.transform(test_data["dialog"])

# %%
with open(upstream['model']['model'], "rb") as rb:
    model = pickle.load(rb)

# %%

test_pred = model.predict(test_x)

accuracy = accuracy_score(test_y, test_pred)

print(accuracy)