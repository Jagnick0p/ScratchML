import pandas as pd

def encoding_categorical(df, ohe_threshold = 10, frq_threshold = 50):
    categorical_cols = df.select_dtypes(include = 'object').columns
    df_encoded = df.copy()
    for col in categorical_cols:
        n_unique = df_encoded[col].nunique()
        if n_unique <= ohe_threshold:
            one_hot_encoding = pd.get_dummies(df_encoded[col], prefix = col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis = 1), one_hot_encoding], axis = 1)

        elif n_unique <= frq_threshold:
            freq_map = df_encoded[col].value_counts().to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)

        else:
            counts = df_encoded[col].value_counts()
            rare_counts = counts[counts < 0.01*len(df_encoded)].index
            df_encoded[col] = df_encoded[col].replace(rare_counts,'other')

            freq_map = df_encoded[col].value_counts().to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)

    return df_encoded

def handling_missing_values(df, drop_threshold = 40):
    df_cleaned = df.copy()
    lower_pct = 5
    missing_pct = df_cleaned.isnull().mean() * 100
    cols_to_impute = missing_pct[(missing_pct > lower_pct) & (missing_pct <= drop_threshold)].index
    cols_for_row_removal = missing_pct[missing_pct < 5].index

    # If missing values are > 40% that blocks are dropped
    df_cleaned = df_cleaned.loc[ : , missing_pct[missing_pct < drop_threshold].index] #This keeps the columns with missing values <40

    # If missing values are < 5% that blocks are dropped
    df_cleaned = df_cleaned.dropna(subset = cols_for_row_removal)

    # If missing values are between 5% - 40% that blocks are dropped
    for col in cols_to_impute:
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    return df_cleaned

def data_scaler(df, numeric_cols, target_col, range_thr = 10):
    df_scaled = df.copy()
    categorical_cols = [col for col in df.columns if col not in numeric_cols + target_col]
    scaling_param = {}
    scaled_cols = []

    for col in numeric_cols:
        if col != target_col:
            mew = df[col].mean()
            sigma = df[col].std()
            df_scaled[col] = (df_scaled[col] - mew) / sigma
            scaling_param[col] = (mew, sigma)
            scaled_cols.append(col)

    for col in categorical_cols:
        unique_values = df_scaled[col].unique()

        if set(unique_values).issubset({0,1}):
            continue

        if df_scaled[col].max() - df_scaled[col].min() > range_thr:
            mew = df[col].mean()
            sigma = df[col].std()
            df_scaled[col] = (df_scaled[col] - mew) / sigma
            scaling_param[col] = (mew, sigma)
            scaled_cols.append(col)
    return df_scaled, scaling_param, scaled_cols