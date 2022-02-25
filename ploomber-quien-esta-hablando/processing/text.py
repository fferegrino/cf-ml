from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from unidecode import unidecode
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import pickle

sp_stopwords = stopwords.words("spanish")
sp_punctuation = string.punctuation + '¿¡'

tk_tokenizer = ToktokTokenizer()
not_wanted = set((unidecode(word) for word in sp_stopwords)) | set(sp_punctuation)

def tokenise(sentence):
    clean = []
    clean_sentence = unidecode(sentence)
    for token_ in tk_tokenizer.tokenize(clean_sentence):
        token = token_.lower()
        if token in not_wanted:
            continue
        clean.append(token)
    return clean

def tokenise_text(upstream, product):
    val = pd.read_csv(upstream["split"]["val"])
    train = pd.read_csv(upstream["split"]["train"])

    dialogs_train = train["dialog"]
    dialogs_val = val["dialog"]

    vectoriser = CountVectorizer(
        binary=True, 
        analyzer=tokenise, 
        max_features=1000)

    train_x = vectoriser.fit_transform(dialogs_train)
    val_x = vectoriser.transform(dialogs_val)

    save_npz(product["train_x"], train_x)
    save_npz(product["val_x"], val_x)

    with open(product["vectoriser"], "wb") as wb:
        pickle.dump(vectoriser, wb)
