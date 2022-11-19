import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin

class CutTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, right=True, labels=None, retbins=False,
                 precision=3, include_lowest=False,
                 duplicates='raise', as_str= True):
        self.bins = bins
        self.right = right
        self.retbins = retbins
        self.precision = precision
        self.include_lowest = include_lowest
        self.duplicates = duplicates
        self.as_str = as_str
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cut_props = deepcopy(self.__dict__)
        as_str = cut_props.pop('as_str')
        for jj in range(X.shape[1]):
            cut_result = pd.cut(x=X.iloc[:, 0].values, **cut_props)
            if as_str:
                X.iloc[:, jj] = cut_result.astype(str).reshape(-1,1)
            else:
                X.iloc[:, jj] = cut_result.reshape(-1,1)
        return X

class YesNoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map = {
            "yes":1,
            "no": 0,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for jj in range(X.shape[1]):
            X_copy[X.columns[jj]] = X.iloc[:, jj].map(self.map)
        return X_copy