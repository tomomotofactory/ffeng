import pandas as pd

from ffeng.util import dtype_is_num, dtype_is_category, test_columns_fit_to_train_columns


class AggregationFEng:
    """
    Aggregation Feature Engineering.

    Convert feature column to aggregation features with id column.
    Aggregation`s key is id value of id column.
    If feature column is numeric, aggregation(min/max/mean/sum/std) feature by id.
    If feature column is category, aggregation(count/rate per id) feature by id.
    """

    def __init__(self):
        self._feature_name: str = None
        self._id_name: str = None
        self._feature_dtype: str = None
        self._feature_columns: pd.Index = None

    def train(self, train_col: pd.Series, train_id_col: pd.Series, feature_name: str, id_name: str = 'id') \
              -> pd.DataFrame:
        """
        Train feature engineering. And convert train_col to features.

        :param train_col: train column to use converting features
        :param train_id_col: id column to use aggregation`s key
        :param feature_name: main name of features
        :param id_name: id name of aggregation feature
        :return: Aggregation features of train_col
        """
        self._feature_name = feature_name
        self._id_name = id_name
        self._feature_dtype = train_col.dtype

        feature_df = self._to_aggregation_feature(train_col, train_id_col)
        self._feature_columns = feature_df.columns
        return feature_df

    def apply(self, test_col: pd.Series, test_id_col: pd.Series) -> pd.DataFrame:
        """
        Apply feature engineering to test_col.
        :param test_col: test column to apply
        :param test_id_col: id column to use aggregation`s key
        :return:  Aggregation features of test_col

        :raise SyntaxError if not call train before
        :raise ValueError if dtype of test_col is not numeric, bool and category
        """
        if self._feature_name is None:
            raise SyntaxError('apply function can use after calling fit function!')

        feature_df = self._to_aggregation_feature(test_col, test_id_col)

        # if categories are not same between train and test, change to columns of train
        if dtype_is_category(self._feature_dtype):
            feature_df = test_columns_fit_to_train_columns(self._feature_columns, feature_df)

        return feature_df

    def _to_aggregation_feature(self, feature_col: pd.Series, id_col: pd.Series) -> pd.DataFrame:

        feature_df = pd.DataFrame({self._id_name: id_col, self._feature_name: feature_col})

        if dtype_is_num(self._feature_dtype):
            # case: feature is num
            grouped_feature_df = feature_df.groupby(self._id_name)\
                .agg({self._feature_name: ['min', 'max', 'mean', 'sum', 'std']})
            grouped_feature_df.replace(pd.np.nan, 0, inplace=True)
            grouped_feature_df.reset_index(inplace=True)
            grouped_feature_df.columns = [self._id_name, self._feature_name + '_min', self._feature_name + '_max',
                                          self._feature_name + '_mean', self._feature_name + '_sum',
                                          self._feature_name + '_std']

            return grouped_feature_df

        elif dtype_is_category(self._feature_dtype):
            # case: feature is category
            id_cnt_df = feature_df.groupby(self._id_name).size()
            id_cnt_df = id_cnt_df.reset_index() # id_cnt_df is Series, so can not use inplace=T

            grouped_cnt_df: pd.Series = feature_df.groupby([self._id_name, self._feature_name]).size()
            grouped_cnt_df = grouped_cnt_df.reset_index() # grouped_cnt_df is Series, so can not use inplace=T
            grouped_cnt_df = pd.merge(grouped_cnt_df, id_cnt_df, on=self._id_name)
            grouped_cnt_df.columns = [self._id_name, self._feature_name, 'count', 'sum_count']
            grouped_cnt_df['rate'] = [float(count) / float(sum_count) for count, sum_count
                                      in zip(grouped_cnt_df['count'], grouped_cnt_df['sum_count'])]

            grouped_feature_df: pd.DataFrame = grouped_cnt_df.pivot(index=self._id_name, columns=self._feature_name,
                                                                    values=['count', 'rate'])
            grouped_feature_df.replace(pd.np.nan, 0, inplace=True)
            grouped_feature_df.columns = [self._feature_name + '_' + str(grouped_feature_df.columns.levels[1][y]) +
                                          '_' + grouped_feature_df.columns.levels[0][x]
                                          for x, y in zip(grouped_feature_df.columns.labels[0],
                                                          grouped_feature_df.columns.labels[1])]
            grouped_feature_df.reset_index(inplace=True)

            return grouped_feature_df

        else:
            raise ValueError('dtype of column must be num or category')
