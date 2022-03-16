import string

from nltk import ToktokTokenizer
from nltk.corpus import stopwords
from unidecode import unidecode

sp_stopwords = stopwords.words("spanish")
sp_punctuation = string.punctuation + "¿¡"

not_wanted = set((unidecode(word) for word in sp_stopwords)) | set(sp_punctuation)

tk_tokenizer = ToktokTokenizer()


def tokenize(sentence):
    clean = []
    clean_sentence = unidecode(sentence)
    for token_ in tk_tokenizer.tokenize(clean_sentence):
        token = token_.lower()
        if token in not_wanted:
            continue
        clean.append(token)
    return clean
