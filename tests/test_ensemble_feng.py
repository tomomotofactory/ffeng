import pytest
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

from ffeng import EnsembleFEng

@pytest.fixture(scope="module", autouse=True)
def log_df():
    df = pd.DataFrame({
        'session_id': ['s1', 's1', 's2', 's3', 's3', 's3', 's3'],
        'buy_price': [100, 50, 30, 0, 10, 20, 80],
        'device': ['pc', 'sp', 'pc', 'sp', 'pc', 'pc', 'tablet'],
        'member_flg': [0, 0, 1, 0, 0, 1, 1],
        'access_count': [1, 1, 3, 1, 0, 3, 2]
    })
    yield df

def test_num_ensemble_feng(log_df):
    feng = EnsembleFEng()

    kfold = KFold(n_splits=2)
    train_feature_df = feng.train(train_df=log_df.loc[:, ['member_flg', 'access_count']],
                                  target_col=log_df.loc[:, 'buy_price'], feature_name='price',
                                  cv_list=kfold.split(log_df), model=XGBRegressor(n_estimators=10),
                                  model_fit_params={'eval_metric': 'rmse'})
    test_feature_df = feng.apply(log_df.loc[:, ['member_flg', 'access_count']])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check

def test_category_ensemble_feng(log_df):
    feng = EnsembleFEng()

    kfold = KFold(n_splits=2)
    log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')

    train_feature_df = feng.train(train_df=log_df.loc[:, ['buy_price', 'access_count']],
                                  target_col=log_df.loc[:, 'member_flg'], feature_name='mflg',
                                  cv_list=kfold.split(log_df), model=LogisticRegression(), model_fit_params={})
    test_feature_df = feng.apply(log_df.loc[:, ['buy_price', 'access_count']])

    print()
    print(train_feature_df.to_string())
    print(test_feature_df.to_string())
    # TODO assert check
