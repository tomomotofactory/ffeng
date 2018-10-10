import pytest
import pandas as pd
from sklearn.model_selection import KFold
from ffeng import KFoldTargetMeanFEng

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

def test_kfold_num_target_mean_feng(log_df):
    feng = KFoldTargetMeanFEng()

    kfold = KFold(n_splits=2)
    log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')

    train_feature_df = feng.train(train_col=log_df.loc[:, 'member_flg'], target_col=log_df.loc[:, 'buy_price'],
                                  feature_name='price', cv_list=kfold.split(log_df))
    test_feature_df = feng.apply(log_df.loc[:, 'member_flg'])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check

def test_kfold_category_target_mean_feng(log_df):
    feng = KFoldTargetMeanFEng()

    kfold = KFold(n_splits=2)
    log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')
    log_df.loc[:, 'device'] = log_df.loc[:, 'device'].astype('category')

    train_feature_df = feng.train(train_col=log_df.loc[:, 'member_flg'], target_col=log_df.loc[:, 'device'],
                                  feature_name='dv', cv_list=kfold.split(log_df))
    test_feature_df = feng.apply(log_df.loc[:, 'member_flg'])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check
