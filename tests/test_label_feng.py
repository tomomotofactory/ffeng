import pytest
import pandas as pd

from ffeng.label_feng import LabelFEng


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

def test_label_feng_case1(log_df):
    feng = LabelFEng()

    log_df.loc[:, 'device'] = log_df.loc[:, 'device'].astype('category')

    train_feature_df = feng.train(log_df.loc[:, 'device'],  'device', category_min_cnt=None, category_min_rate=None,
                                  aggregation_value = 'other_category')
    test_feature_df = feng.apply(log_df.loc[:, 'device'])

    assert train_feature_df.shape[0] == 7
    assert train_feature_df.shape[1] == 3
    assert test_feature_df.shape[0] == 7
    assert test_feature_df.shape[1] == 3

    assert train_feature_df.iloc[1,1] == 1
    assert train_feature_df.iloc[6,2] == 1

    assert test_feature_df.iloc[1,1] == 1
    assert test_feature_df.iloc[6,2] == 1

def test_label_feng_case2(log_df):
    feng = LabelFEng()

    train_feature_df = feng.train(log_df.loc[1:4, 'device'],  'device', category_min_cnt=None, category_min_rate=None,
                                  aggregation_value = 'other_category')
    test_feature_df = feng.apply(log_df.loc[5:6, 'device'])

    assert train_feature_df.shape[1] == 2
    assert test_feature_df.shape[1] == 2

def test_label_feng_case3(log_df):
    feng = LabelFEng()

    train_feature_df = feng.train(log_df.loc[:, 'device'],  'device',
                                  category_min_cnt=3, category_min_rate=None, aggregation_value = 'others')
    test_feature_df = feng.apply(log_df.loc[0:1, 'device'])

    assert train_feature_df.shape[1] == 2
    assert test_feature_df.shape[1] == 2
    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check

