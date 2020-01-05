def join_features(df, feature_fns):
    ret_df = df
    for feature_fn in feature_fns:
        ret_df = ret_df.join(feature_fn(df))
    return ret_df
