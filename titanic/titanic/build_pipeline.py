import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import clean_data

def build_pipeline():
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', build_preprocessor()),
                                  ('model', define_model())
                                 ])
    return my_pipeline

def evaluate_pipeline(pipeline, X_train, y_train, X_valid, y_valid):
    # Preprocessing of training data, fit model
    pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds_df = make_predictions(pipeline, X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds_df.values)
    print('MAE:', score)

    num_equal = len(y_valid[preds_df['Survived'] == y_valid['Survived']])
    pct_equal = float(num_equal) / float(len(y_valid))
    print('Percent correct pred:', pct_equal)

    return preds_df

def make_predictions(pipeline, df):
    preds = pipeline.predict(df)
    return pd.DataFrame(
        preds,
        index=df.index,
        columns=['Survived']
    )

def define_model():
    return LogisticRegression()

def build_preprocessor():
    numerical_cols = clean_data.numerical_features()
    categorical_cols = clean_data.categorical_features()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor
