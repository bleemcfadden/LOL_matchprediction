import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer


#patch
def get_patch_major(x):
    try:
        return int(str(x).split(".")[0])
    except:
        return 0


#token making
def make_tokens(df, role_cols, ban_cols):
    tokens = []
    for _, r in df.iterrows():
        t = []
        for c in role_cols:
            if pd.notna(r[c]):
                t.append(f"HAS_{r[c]}")
        for c in ban_cols:
            if pd.notna(r[c]):
                t.append(f"BAN_{r[c]}")
        tokens.append(t)
    return tokens

def encode_tokens(train_df, test_df, role_cols, ban_cols):
    mlb = MultiLabelBinarizer(sparse_output=True)
    Xtr_tok = mlb.fit_transform(make_tokens(train_df, role_cols, ban_cols))
    Xte_tok = mlb.transform(make_tokens(test_df, role_cols, ban_cols))
    return Xtr_tok, Xte_tok, mlb

#feature transformers
class LolPOVFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, role_cols, ban_cols, cat_cols, num_cols):
        self.role_cols = role_cols
        self.ban_cols = ban_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.mlb_ = None
        self.ohe_ = None
        self.scaler_ = None

    def fit(self, X, y=None):
        df = X.copy()

        df["patch_major"] = df["patch"].apply(get_patch_major)

        # tokens
        self.mlb_ = MultiLabelBinarizer(sparse_output=True)
        X_tok = self.mlb_.fit_transform(
            make_tokens(df, self.role_cols, self.ban_cols)
        )

        # categorical
        for c in self.cat_cols:
            df[c] = df[c].astype("string").fillna("Unknown")

        if len(self.cat_cols) > 0:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            X_cat = self.ohe_.fit_transform(df[self.cat_cols])
        else:
            self.ohe_ = None

        # numeric
        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        self.scaler_ = StandardScaler()
        X_num = csr_matrix(self.scaler_.fit_transform(num))

        return self

    def transform(self, X):
        df = X.copy()

        df["patch_major"] = df["patch"].apply(get_patch_major)

        X_tok = self.mlb_.transform(
            make_tokens(df, self.role_cols, self.ban_cols)
        )

        if self.ohe_ is not None:
            X_cat = self.ohe_.transform(
                df[self.cat_cols].astype("string").fillna("Unknown")
            )
            cat_block = [X_cat]
        else:
            cat_block = []

        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        X_num = csr_matrix(self.scaler_.transform(num))

        return hstack([X_tok, *cat_block, X_num], format="csr").tocsr()