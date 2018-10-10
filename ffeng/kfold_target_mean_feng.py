from typing import Sequence, List
import pandas as pd

from ffeng.util import dtype_is_num, dtype_is_category


class KFoldTargetMeanFEng:
    """
    KFold Target Mean Feature Engineering.
    (This feature engineering is very similar to ensemble feature engineering, but don`t use predict model.)

    Convert category feature column to KFold target mean features.
    """

    def __init__(self):
        self._feature_name: str = None
        self._target_dtype: str = None
        self._category_target_mst: pd.DataFrame = None

    def train(self, train_col: pd.Series, target_col: pd.Series, feature_name: str, cv_list: Sequence[Sequence[int]])\
              -> pd.DataFrame:
        """
        Train feature engineering. And convert train_col to features.
        Cache target mean master data calculated by train_df and target_col.

        :param train_col: train column to use converting features
        :param target_col: target column to calculate mean
        :param feature_name: main name of features
        :param cv_list: cross-validation (Sequence[train_index, test_index])
        :return: KFold Target Mean features of train_df

        :raise ValueError if dtype of test_col is not numeric, bool and category
        """
        self._feature_name = feature_name
        self._target_dtype = target_col.dtype
        feature_df = pd.DataFrame({'category_val': train_col, 'target': target_col})

        # get unique category values for check cv data has all categories
        feature_df.loc[:, 'category_val'] = feature_df.loc[:, 'category_val'].astype('category')
        uq_category_vals = feature_df.loc[:, 'category_val'].cat.categories

        if dtype_is_num(self._target_dtype):
            # case: target is num

            train_df = None
            for train_index, test_index in cv_list:
                category_target_mst = KFoldTargetMeanFEng._to_category_target_mean_df(feature_df.iloc[train_index, :],
                                                                                      feature_name, uq_category_vals)

                tmp_train_df= pd.merge(feature_df.iloc[test_index, :].loc[:, ['category_val']], category_target_mst,
                                       how='inner', on='category_val')
                tmp_train_df.drop(columns='category_val', inplace=True)
                tmp_train_df.index = test_index

                if train_df is None:
                    train_df = tmp_train_df
                else:
                    train_df = pd.concat([train_df, tmp_train_df], axis=0)

            # For test_col: We can use all data for calculating mean of target
            self._category_target_mst = KFoldTargetMeanFEng._to_category_target_mean_df(feature_df, feature_name)

            return train_df

        elif dtype_is_category(self._target_dtype):
            # case: target is category

            train_df = None
            for train_index, test_index in cv_list:
                category_target_proba_mst = \
                    KFoldTargetMeanFEng._to_category_target_proba_dict(feature_df.iloc[train_index, :],
                                                                       feature_name, uq_category_vals)

                tmp_train_df = pd.merge(feature_df.iloc[test_index, :].loc[:, ['category_val']],
                                        category_target_proba_mst, how='inner', on='category_val')
                tmp_train_df.drop(columns='category_val', inplace=True)
                tmp_train_df.index = test_index

                if train_df is None:
                    train_df = tmp_train_df
                else:
                    train_df = pd.concat([train_df, tmp_train_df], axis=0, sort=False)

            train_df.replace(pd.np.nan, 0, inplace=True)

            # For test_col: We can use all data for calculating probability of target
            self._category_target_mst = KFoldTargetMeanFEng._to_category_target_proba_dict(feature_df, feature_name)
            return train_df

        else:
            raise ValueError('dtype of target is wrong.')

    def apply(self, test_col: pd.Series) -> pd.DataFrame:
        """
        Apply feature engineering to test_col by cached target mean master data.

        :param test_col: test column to apply
        :return: KFold Target Mean features of test_col

        :raise SyntaxError if not call train before
        """
        if self._feature_name is None:
            raise SyntaxError('apply function can use after calling fit function!')

        if dtype_is_num(self._target_dtype):
            test_df = pd.merge(test_col.to_frame(name='category_val'), self._category_target_mst,
                               how='left', on='category_val')
            test_df.replace(pd.np.nan, 0, inplace=True)
            test_df.drop(columns='category_val', inplace=True)
            return test_df

        elif dtype_is_category(self._target_dtype):
            test_df = pd.merge(test_col.to_frame(name='category_val'), self._category_target_mst,
                               how='left', on='category_val')
            test_df.replace(pd.np.nan, 0, inplace=True)
            test_df.drop(columns='category_val', inplace=True)
            return test_df

        else:
            # never happen
            raise ValueError('dtype of target_col is num or category')

    @staticmethod
    def _to_category_target_mean_df(feature_df: pd.DataFrame, feature_name:str, uq_category_vals: Sequence=None)\
            -> pd.DataFrame:
        category_target_mst = feature_df.groupby('category_val').agg({'target': 'mean'})

        # check for using cv case
        if uq_category_vals is not None:
            for uq_category_val in uq_category_vals:
                if uq_category_val not in category_target_mst.index:
                    raise ValueError('Not include data of category value:' + uq_category_val + ' in CVData.')

        category_target_mst.rename(columns={'target': feature_name + '_mean'}, inplace=True)

        return category_target_mst

    @staticmethod
    def _to_category_target_proba_dict(feature_df: pd.DataFrame, feature_name:str, uq_category_vals: Sequence=None)\
            -> pd.DataFrame:
        category_proba_array: List[Sequence] = []

        category_mst_dict = dict(feature_df.groupby(['category_val']).size())

        # check for using cv case
        if uq_category_vals is not None:
            for uq_category_val in uq_category_vals:
                if uq_category_val not in category_mst_dict.keys():
                    raise ValueError('Not include data of category value:' + str(uq_category_val) + ' in CVData.')

        category_target_mst = feature_df.groupby(['category_val', 'target']).size()
        for category_val_index_no in range(max(category_target_mst.index.labels[0]) + 1):
            category_val = category_target_mst.index.levels[0][category_val_index_no]
            proba_array: List = [category_val]

            for target_index_no in range(max(category_target_mst.index.labels[1]) + 1):
                target_val = category_target_mst.index.levels[1][target_index_no]

                if (category_val, target_val) in category_target_mst.index:
                    size = category_target_mst.loc[category_val, target_val, :].values[0]
                    proba = float(size) / float(category_mst_dict[category_val])
                else:
                    # case cnt = 0
                    proba = 0

                proba_array.append(proba)

            category_proba_array.append(proba_array)

        column_names = ['category_val']
        column_names.extend([feature_name + '_' + str(category_target_mst.index.levels[1][x]) + '_probability'
                             for x in range(max(category_target_mst.index.labels[1]) + 1)])

        mst_df = pd.DataFrame(data=category_proba_array, columns=column_names)
        mst_df.loc[:, ['category_val']] = mst_df.loc[:, ['category_val']].astype('category')

        return mst_df
