# fast_feng
Feature engineering library for python.
This library help for converting to bin counting feature, ensemble feature or other technique.
This library ensures that the results of feature engineering on the training data and test data have the same columns.

## Requirement
- Python 3.6 or later
- pandas 0.23.4 or later

## Install

### Use setup.py
```shell
# Master branch
$ git clone https://github.com/tomomotofactory/ffeng.git
$ python setup.py install
```

### Use pip
```shell
# Master branch
$ pip install git+https://github.com/tomomotofactory/ffeng.git
# Specific tag (or branch, commit hash)
$ pip install git+https://github.com/tomomotofactory/ffeng@v0.1.0
```

## How to use

### Load sample data
```python
import pandas as pd

log_df = pd.DataFrame({
    'session_id': ['s1', 's1', 's2', 's3', 's3', 's3', 's3'],
    'buy_price': [100, 50, 30, 0, 10, 20, 80],
    'device': ['pc', 'sp', 'pc', 'sp', 'pc', 'pc', 'tablet'],
    'member_flg': [0, 0, 1, 0, 0, 1, 1],
    'access_count': [1, 1, 3, 1, 0, 3, 2]
})
```

### Label Feature Engineering
```python
from ffeng import LabelFEng

feng = LabelFEng()

train_feature_df = feng.train(train_col=log_df.loc[:, 'device'],  feature_name='device',
                              category_min_cnt=3, category_min_rate=None, aggregation_value = 'others')
test_feature_df = feng.apply(log_df.loc[0:1, 'device'])

print(train_feature_df.to_string())
"""
   device_others  device_pc
0              0          1
1              1          0
2              0          1
3              1          0
4              0          1
5              0          1
6              1          0
"""

print(test_feature_df.to_string())
"""
   device_others  device_pc
0              0          1
1              1          0
"""
```

### Bin Counting Feature Engineering
```python
from ffeng import BinCountingFEng

feng = BinCountingFEng()

train_feature_df = feng.train(log_df.loc[:4, 'device'], 'dv')
test_feature_df = feng.apply(log_df.loc[5:, 'device'])

print(train_feature_df.to_string())
"""
   dv_count
0         3
1         2
2         3
3         2
4         3
"""

print(test_feature_df.to_string())
"""
   dv_count
0         3
1         0
"""
```

### KFold target mean Feature Engineering (num)
```python
from sklearn.model_selection import KFold
from ffeng import KFoldTargetMeanFEng

feng = KFoldTargetMeanFEng()

kfold = KFold(n_splits=2)
log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')

train_feature_df = feng.train(train_col=log_df.loc[:, 'member_flg'], target_col=log_df.loc[:, 'buy_price'],
                              feature_name='price', cv_list=kfold.split(log_df))
test_feature_df = feng.apply(log_df.loc[:, 'member_flg'])

print(train_feature_df.to_string())
"""
   price_mean
0          10
1          10
2          10
3          50
4          50
5          30
6          30
"""

print(test_feature_df.to_string())
"""
   price_mean
0   40.000000
1   40.000000
2   43.333333
3   40.000000
4   40.000000
5   43.333333
6   43.333333
"""
```

### KFold target mean Feature Engineering (category)
```python
from sklearn.model_selection import KFold
from ffeng import KFoldTargetMeanFEng

feng = KFoldTargetMeanFEng()

kfold = KFold(n_splits=2)
log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')
log_df.loc[:, 'device'] = log_df.loc[:, 'device'].astype('category')

train_feature_df = feng.train(train_col=log_df.loc[:, 'member_flg'], target_col=log_df.loc[:, 'device'],
                              feature_name='dv', cv_list=kfold.split(log_df))
test_feature_df = feng.apply(log_df.loc[:, 'member_flg'])

print(train_feature_df.to_string())
"""
   dv_pc_probability  dv_sp_probability  dv_tablet_probability
0           1.000000           0.000000                    0.0
1           1.000000           0.000000                    0.0
2           1.000000           0.000000                    0.0
3           0.500000           0.000000                    0.5
4           0.333333           0.666667                    0.0
5           1.000000           0.000000                    0.0
6           1.000000           0.000000                    0.0
"""

print(test_feature_df.to_string())
"""
   dv_pc_probability  dv_sp_probability  dv_tablet_probability
0           0.500000                0.5               0.000000
1           0.500000                0.5               0.000000
2           0.666667                0.0               0.333333
3           0.500000                0.5               0.000000
4           0.500000                0.5               0.000000
5           0.666667                0.0               0.333333
6           0.666667                0.0               0.333333
"""
```

### Ensemble Feature Engineering (predict:num)
```python
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from ffeng import EnsembleFEng

feng = EnsembleFEng()

kfold = KFold(n_splits=2)

train_feature_df = feng.train(train_df=log_df.loc[:, ['member_flg', 'access_count']],
                              target_col=log_df.loc[:, 'buy_price'], feature_name='price',
                              cv_list=kfold.split(log_df), model=XGBRegressor(n_estimators=10),
                              model_fit_params={'eval_metric': 'rmse'})
test_feature_df = feng.apply(log_df.loc[:, ['member_flg', 'access_count']])

print(train_feature_df.to_string())
"""
       price
0   4.311999
1   4.311999
2   8.324629
3   4.311999
4  26.591251
5  20.195786
6  20.195786
"""

print(test_feature_df.to_string())
"""
       price
0  26.591251
1  26.591251
2  20.195786
3  26.591251
4  26.591251
5  20.195786
6  20.195786
"""
```

### Ensemble Feature Engineering (predict:category)
```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from ffeng import EnsembleFEng

feng = EnsembleFEng()

kfold = KFold(n_splits=2)
log_df.loc[:, 'member_flg'] = log_df.loc[:, 'member_flg'].astype('bool')

train_feature_df = feng.train(train_df=log_df.loc[:, ['buy_price', 'access_count']],
                              target_col=log_df.loc[:, 'member_flg'], feature_name='mflg',
                              cv_list=kfold.split(log_df), model=LogisticRegression(), model_fit_params={})
test_feature_df = feng.apply(log_df.loc[:, ['buy_price', 'access_count']])

print(train_feature_df.to_string())
"""
   mflg_is_False  mflg_is_True
0       0.022659      0.977341
1       0.124318      0.875682
2       0.101481      0.898519
3       0.465050      0.534950
4       0.662674      0.337326
5       0.355547      0.644453
6       0.878088      0.121912
"""

print(test_feature_df.to_string())
"""
   mflg_is_False  mflg_is_True
0       0.743736      0.256264
1       0.608881      0.391119
2       0.183656      0.816344
3       0.455057      0.544943
4       0.687173      0.312827
5       0.165709      0.834291
6       0.493429      0.506571
"""
```

### Aggregation Feature Engineering (num)
```python
from ffeng import AggregationFEng

feng = AggregationFEng()

train_feature_df = feng.train(train_col=log_df.loc[:, 'buy_price'], train_id_col=log_df.loc[:, 'session_id'],
                              feature_name='buy_price', id_name='session_id')
test_feature_df = feng.apply(test_col=log_df.loc[1:2, 'buy_price'], test_id_col=log_df.loc[1:2, 'session_id'])


print(train_feature_df.to_string())
"""
  session_id  price_min  price_max  price_mean  price_sum  price_std
0         s1         50        100        75.0        150  35.355339
1         s2         30         30        30.0         30   0.000000
2         s3          0         80        27.5        110  35.939764
"""

print(test_feature_df.to_string())
"""
  session_id  price_min  price_max  price_mean  price_sum  price_std
0         s1         50         50          50         50        0.0
1         s2         30         30          30         30        0.0
"""
```

### Aggregation Feature Engineering (category)
```python
from ffeng import AggregationFEng

feng = AggregationFEng()

log_df.loc[:, 'device'] = log_df.loc[:, 'device'].astype('category')

train_feature_df = feng.train(train_col=log_df.loc[:, 'device'], train_id_col=log_df.loc[:, 'session_id'],
                              feature_name='dv', id_name='session_id')
test_feature_df = feng.apply(test_col=log_df.loc[1:2, 'device'], test_id_col=log_df.loc[1:2, 'session_id'])

print(train_feature_df.to_string())
"""
  session_id  dv_pc_count  dv_sp_count  dv_tablet_count  dv_pc_rate  dv_sp_rate  dv_tablet_rate
0         s1          1.0          1.0              0.0         0.5        0.50            0.00
1         s2          1.0          0.0              0.0         1.0        0.00            0.00
2         s3          2.0          1.0              1.0         0.5        0.25            0.25
"""

print(test_feature_df.to_string())
"""
  session_id  dv_pc_count  dv_sp_count  dv_tablet_count  dv_pc_rate  dv_sp_rate  dv_tablet_rate
0         s1          0.0          1.0                0         0.0         1.0               0
1         s2          1.0          0.0                0         1.0         0.0               0
"""
```