import pandas as pd


class BinCountingFEng:
    """
    Bin Counting Feature Engineering.

    Convert category feature column to bin counting features.
    """

    def __init__(self):
        self._category_cnt_mst: pd.DataFrame = None
        self._feature_name: str = None

    def train(self, train_col: pd.Series, feature_name: str) -> pd.DataFrame:
        """
        Train feature engineering. And convert train_col to features.
        Cache counting master data for converting to bin counting features.

        :param train_col: train column to use converting features
        :param feature_name: main name of features
        :return: Bin counting features of train_col
        """
        self._feature_name = feature_name
        category_cnt_series: pd.Series = train_col.value_counts()
        self._category_cnt_mst = pd.DataFrame({self._feature_name: category_cnt_series.index,
                                               self._feature_name + '_count': category_cnt_series.values})
        return self._to_cnt_features(train_col)

    def apply(self, test_col: pd.Series):
        """
        Apply feature engineering to test_col by cached counting master data.

        :param test_col: test column to apply
        :return: Aggregation features of test_col

        :raise SyntaxError if not call train before
        """
        if self._feature_name is None:
            raise SyntaxError('apply function can use after calling fit function!')

        return self._to_cnt_features(test_col)

    def _to_cnt_features(self, feature_col: pd.Series) -> pd.DataFrame:
        feature_df = pd.merge(feature_col.to_frame(name=self._feature_name), self._category_cnt_mst,
                              on=self._feature_name, how='left')
        feature_df.replace(pd.np.nan, int(0), inplace=True)
        feature_df.iloc[:, 1] = feature_df.iloc[:, 1].astype('int')

        feature_df.drop(columns=self._feature_name, inplace=True)

        return feature_df