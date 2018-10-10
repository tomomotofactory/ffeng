import pytest
import pandas as pd

from ffeng import BinCountingFEng

@pytest.fixture(scope="module", autouse=True)
def log_df():
    df = pd.DataFrame({
        'session_id': ['s1', 's1', 's2', 's3', 's3', 's3', 's3'],
        'buy_price': [100, 50, 30, 0, 10, 20, 80],
        'device': ['pc', 'sp', 'pc', 'sp', 'pc', 'pc', 'tablet'],
        'member_flg': [0, 0, 1, 0, 0, 1, 1],
        'access_count': [1, 2, 1, 1, 2, 3, 4]
    })
    yield df

def test_bin_count_feng_case1(log_df):
    feng = BinCountingFEng()

    train_feature_df = feng.train(log_df.loc[:4, 'device'], 'dv')
    test_feature_df = feng.apply(log_df.loc[5:, 'device'])

    print()
    print(train_feature_df)
    print(test_feature_df)
    # TODO assert check

def test_bin_count_feng_case2(log_df):
    feng = BinCountingFEng()

    train_feature_df = feng.train(log_df.loc[:1, 'device'], 'device_cnt')
    test_feature_df = feng.apply(log_df.loc[:, 'device'])

    print()
    print(train_feature_df)
    print(test_feature_df)
    # TODO assert check