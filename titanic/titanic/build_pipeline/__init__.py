import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import preprocessors
from . import models


def build_default_pipeline():
    return build_pipeline(preprocessors.build_preprocessor, models.define_model)


def build_pipeline(preprocessor_fn, model_fn):
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor_fn()), ("model", model_fn())]
    )
    return my_pipeline


def evaluate_pipeline(pipeline, X_train, y_train, X_valid, y_valid):
    # Preprocessing of training data, fit model
    pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds_df = make_predictions(pipeline, X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds_df.values)
    print("MAE:", score)

    num_equal = len(y_valid[preds_df["Survived"] == y_valid["Survived"]])
    pct_equal = float(num_equal) / float(len(y_valid))
    print("Percent correct pred:", pct_equal)

    return preds_df


def make_predictions(pipeline, df):
    preds = pipeline.predict(df)
    return pd.DataFrame(preds, index=df.index, columns=["Survived"])
