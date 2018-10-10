import pandas as pd


def dtype_is_num(dtype: str) -> bool:
    return dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                     'float16', 'float32', 'float64', 'float128']

def dtype_is_category(dtype: str) -> bool:
    return str(dtype) in ['bool', 'category']

def test_columns_fit_to_train_columns(train_columns: pd.Index, test_feature_df: pd.DataFrame):
    test_feature_df = pd.concat([pd.DataFrame(columns=train_columns), test_feature_df], sort=False)
    test_feature_df = test_feature_df.loc[:, train_columns]
    test_feature_df.replace(pd.np.nan, 0, inplace=True)

    return test_feature_df
