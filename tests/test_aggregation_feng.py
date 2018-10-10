import pytest
import pandas as pd

from ffeng import AggregationFEng

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

def test_aggregation_feng_num_case1(log_df):
    feng = AggregationFEng()

    train_feature_df = feng.train(train_col=log_df.loc[:, 'buy_price'], train_id_col=log_df.loc[:, 'session_id'],
                                  feature_name='price', id_name='session_id')
    test_feature_df = feng.apply(test_col=log_df.loc[1:2, 'buy_price'], test_id_col=log_df.loc[1:2, 'session_id'])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check

def test_aggregation_feng_category_case2(log_df):
    feng = AggregationFEng()

    log_df.loc[:, 'device'] = log_df.loc[:, 'device'].astype('category')

    train_feature_df = feng.train(train_col=log_df.loc[:, 'device'], train_id_col=log_df.loc[:, 'session_id'],
                                  feature_name='dv', id_name='session_id')
    test_feature_df = feng.apply(test_col=log_df.loc[1:2, 'device'], test_id_col=log_df.loc[1:2, 'session_id'])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check