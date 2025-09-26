import pandas as pd

def encoding_categorical(df, ohe_threshold = 10, frq_threshold = 50):
    categorical_cols = df.select_dtypes(include = 'object').columns
    df_encoded = df.copy()
    for col in categorical_cols:
        n_unique = df_encoded[col].nunique()
        if n_unique <= ohe_threshold:
            one_hot_encoding = pd.get_dummies(df_encoded[col], prefix = col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis = 1), one_hot_encoding], axis = 1)

        elif n_unique <= ohe_threshold:
            freq_map = df_encoded[col].value_counts().to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)

        else:
            counts = df_encoded[col].value_counts()
            rare_counts = counts[counts < 0.01*len(df_encoded)].index
            df_encoded[col] = df_encoded[col].replace(rare_counts,'other')


