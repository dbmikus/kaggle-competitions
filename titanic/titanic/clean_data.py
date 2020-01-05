from sklearn.model_selection import train_test_split

from .lib import features as features_lib
from . import build_features

def prepare_dataframes(train_df, test_df):
    feature_fns = [
        build_features.split_ticket,
        build_features.cabin_character,
    ]
    train_df = features_lib.join_features(train_df, feature_fns)
    test_df = features_lib.join_features(test_df, feature_fns)

    clean_train_df = clean_df(train_df)
    clean_test_df = clean_df(test_df)

    X_train, X_valid, y_train, y_valid = _split_train_x_y_df(clean_train_df)

    return {
        # uncleaned, but with new features
        'uncleaned': {
            'train': train_df,
            'test': test_df,
        },
        # cleaned, with new features, but not split
        'cleaned': {
            'train': clean_train_df,
            'test': clean_test_df,
        },

        # The following are: cleaned, with new features, and split
        'train': (X_train, y_train),
        'validation': (X_valid, y_valid),
        'test': (clean_test_df, None),
    }

def clean_df(df):
    return df.drop(
        columns=[
            'Ticket',
            'Name',

            # TODO add these back in later

            # Look at `set(clean_df['ticket_prefix'])` to see ticket prefixes
            # that might be similar
            'ticket_prefix',
        ]
    )

def categorical_features():
    return [
        'Sex',
        'Embarked',
        # TODO should this be converted to an ordered numerical value?
        'cabin_char',
    ]

def numerical_features():
    return [
        'PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'
    ]

def _split_train_x_y_df(cleaned_train_df, valid_size=0.33):
    y_col_name = 'Survived'
    X = cleaned_train_df.drop(columns=[y_col_name])
    y = cleaned_train_df[[y_col_name]]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size,
    )
    return X_train, X_valid, y_train, y_valid
