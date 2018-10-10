import pandas as pd

from ffeng.util import test_columns_fit_to_train_columns


class LabelFEng:
    """
    Label Feature Engineering.

    Convert category feature column to dummy label features and combine small category.
    """

    def __init__(self):
        self._feature_dtype: str = None
        self._feature_name: str = None
        self._feature_columns: pd.Index = None
        self._aggregation_value: str = None

    def train(self, train_col: pd.Series, feature_name: str, category_min_cnt: int = None,
              category_min_rate: float = None, aggregation_value = 'others') -> pd.DataFrame:
        """
        Train feature engineering. And convert train_col to features.
        Cache categories in train_col.
        If set category_min_cnt or/and category_min_rate, small category value replace to aggregation_value.

        :param train_col: train column to use converting features
        :param feature_name: main name of features
        :param category_min_cnt: minimum count of category to keep category value
        :param category_min_rate: minimum rate of category to keep category value
        :param aggregation_value: name of aggregation value
        :return: Label features of train_col
        """
        self._feature_dtype = train_col.dtype
        self._feature_name = feature_name
        self._aggregation_value = aggregation_value

        feature_df = self._to_label_features_with_min_cut(train_col, feature_name, category_min_cnt,
                                                          category_min_rate, aggregation_value)

        return feature_df

    def apply(self, test_col: pd.Series) -> pd.DataFrame:
        """
        Apply feature engineering to test_col by cached categories.

        :param test_col: test column to use converting features
        :return: Label features of train_col
        """
        if self._feature_name is None:
            raise SyntaxError('apply function can use after calling fit function!')

        # TODO check dtype
        feature_df = self._to_label_features(test_col)

        # if categories are not same between train and test, change to columns of train
        feature_df = test_columns_fit_to_train_columns(self._feature_columns, feature_df)

        return feature_df

    def _to_label_features_with_min_cut(self, feature_col: pd.Series, feature_name: str, category_min_cnt: int = None,
                                        category_min_rate: float = None, aggregation_value = 'others') -> pd.DataFrame:
        feature_col: pd.Series = feature_col.astype('object')

        if category_min_cnt is not None or category_min_rate is not None:
            category_cnt_series: pd.Series = feature_col.value_counts()
            sum_cnt = len(feature_col.values)

            for category_val, cnt in zip(category_cnt_series.index, category_cnt_series.values):
                if (category_min_cnt is not None and cnt < category_min_cnt) \
                        or (category_min_rate is not None and float(cnt) / float(sum_cnt) < category_min_rate):
                    feature_col.replace(category_val, aggregation_value, inplace=True)

        feature_col: pd.Series = feature_col.astype('category')
        feature_df = feature_col.to_frame(name=feature_name)

        dummy_df: pd.DataFrame = pd.get_dummies(feature_df, drop_first=False)
        self._feature_columns = dummy_df.columns

        return dummy_df

    def _to_label_features(self, feature_col: pd.Series) -> pd.DataFrame:
        feature_col: pd.Series = feature_col.astype('category')
        uq_vals = feature_col.cat.categories
        feature_col: pd.Series = feature_col.astype('object')

        for category_val in uq_vals:
            if self._feature_name + '_' +category_val not in self._feature_columns.values:
                feature_col.replace(category_val, self._aggregation_value, inplace=True)

        feature_col: pd.Series = feature_col.astype('category')
        feature_df = feature_col.to_frame(name=self._feature_name)

        return pd.get_dummies(feature_df, drop_first=False)