import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from itertools import combinations
import joblib


# ---------- Utility Function ----------
def create_categorical_combinations(X, r_list, cat_cols):
    df_str = X[cat_cols].astype(str)
    for r in r_list:
        for comb in combinations(cat_cols, r):
            df_str["+".join(comb)] = df_str[list(comb)].agg("".join, axis=1)
    return df_str.drop(columns=cat_cols)


# ---------- Custom Transformer ----------
class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.membership_order = [
            "No Membership",
            "Basic Membership",
            "Silver Membership",
            "Gold Membership",
            "Platinum Membership",
            "Premium Membership",
        ]
        self.positive_feedback = [
            "Products always in Stock",
            "Quality Customer Care",
            "Reasonable Price",
            "User Friendly Website",
        ]
        self.negative_feedback = [
            "Poor Website",
            "Poor Customer Service",
            "Poor Product Quality",
            "Too many ads",
        ]

    def time_of_day(self, hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    def ampm_mapping(self, hour):
        return "AM" if 0 <= hour < 12 else "PM"

    def get_sentiment(self, feedback):
        if feedback in self.positive_feedback:
            return 1
        elif feedback in self.negative_feedback:
            return -1
        else:
            return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["points_per_transaction"] = X["points_in_wallet"] / X["avg_transaction_value"]
        X["transaction_value_per_time_unit"] = (
            X["avg_transaction_value"] / X["avg_time_spent"]
        )

        X["last_visit_hour"] = pd.to_datetime(X["last_visit_time"]).dt.hour
        X["last_visit_time_of_day"] = X["last_visit_hour"].apply(self.time_of_day)
        X["last_visit_AMPM"] = X["last_visit_hour"].apply(self.ampm_mapping)
        X.drop("last_visit_time", axis=1, inplace=True)

        X["joining_date"] = pd.to_datetime(X["joining_date"])
        X["joining_day_name"] = X["joining_date"].dt.day_name()
        X["is_weekend"] = X["joining_day_name"].isin(["Saturday", "Sunday"]).astype(int)
        X.drop("joining_date", axis=1, inplace=True)

        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        cat_combos_df = create_categorical_combinations(X, [2], cat_cols)
        X = pd.concat([X, cat_combos_df], axis=1)

        X["membership_category"] = pd.Categorical(
            X["membership_category"], categories=self.membership_order, ordered=True
        ).codes
        X["feedback"] = X["feedback"].apply(self.get_sentiment)
        X["avg_time_spent_log"] = np.log1p(X["avg_time_spent"])
        X["avg_transaction_value_square"] = np.square(X["avg_transaction_value"])
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# ---------- Columns to Scale ----------
scale_cols = [
    "age",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "points_per_transaction",
    "transaction_value_per_time_unit",
    "last_visit_hour",
    "avg_transaction_value_square",
]

# ---------- Encoder and Scaler Transformer ----------
encoder_scaler_transformer = ColumnTransformer(
    [
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int),
            make_column_selector(dtype_include=["object"]),
        ),
        ("scaler", StandardScaler(), scale_cols),
    ],
    remainder="passthrough",
)

# ---------- Full Pipeline ----------
featureEng_encoder_scaler_pipeline = Pipeline(
    [("featureEng", FeatureEng()), ("encoder_scaler", encoder_scaler_transformer)]
)

# ---------- Optional: Save Pipeline ----------
# joblib.dump(featureEng_encoder_scaler_pipeline, 'feature_eng_pipeline.pkl')
