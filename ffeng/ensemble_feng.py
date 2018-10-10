from typing import Sequence, Dict
import pandas as pd

from ffeng.util import dtype_is_num, dtype_is_category


class EnsembleFEng:
    """
    Ensemble Feature Engineering.

    Convert category feature column to ensemble features.
    """

    def __init__(self):
        self._feature_name: str = None
        self._target_dtype: str = None
        self._model = None

    def train(self, train_df: pd.DataFrame, target_col: pd.Series, feature_name:str, cv_list: Sequence[Sequence[int]],
              model,  model_fit_params: Dict, cv_model_fit_params: Dict = None) \
              -> pd.DataFrame:
        """
        Train feature engineering. And convert train_col to features.
        Cache predict model of training by train_df and target_col.

        :param train_df: train dataframe to use converting ensemble features and training model
        :param target_col: target column to use training model
        :param feature_name: main name of features
        :param cv_list: cross-validation (Sequence[train_index, test_index])
        :param model: model to use ensemble
        :param model_fit_params: parameters for model.fit
        :param cv_model_fit_params: parameters for model.fit in cross-validation
        :return: Ensemble features of train_df

        :raise ValueError if all categories of target not include in cross-validation training data
        :raise ValueError if dtype of target_col is not numeric, bool and category
        """
        self._feature_name = feature_name
        self._target_dtype = target_col.dtype
        self._model = model

        # if not set cv_model_fit_params, cv_model_fit_params use model_fit_params
        if cv_model_fit_params is None:
            cv_model_fit_params = model_fit_params

        if dtype_is_num(self._target_dtype):
            # case: target dtype is num
            self._model.fit(train_df, target_col, **model_fit_params)

            train_predicts: pd.Series = None
            for train_index, test_index in cv_list:
                self._model.fit(train_df.iloc[train_index, :], target_col[train_index], **cv_model_fit_params)
                tmp_train_predict_df = pd.Series(self._model.predict(train_df.iloc[test_index, :]), index=test_index)

                if train_predicts is None:
                    train_predicts = tmp_train_predict_df
                else:
                    train_predicts = train_predicts.append(tmp_train_predict_df)

            return train_predicts.to_frame(feature_name)

        elif dtype_is_category(self._target_dtype):
            # case: target dtype is category
            if 'bool' == self._target_dtype:
                uq_target_vals = [True, False]
            else:
                uq_target_vals = target_col.cat.categories
            train_predict_df = None

            # TODO Parallel processing
            for train_index, test_index in cv_list:

                # check for using cv case
                uq_train_target_vals = target_col[train_index].unique()
                for uq_target_val in uq_target_vals:
                    if uq_target_val not in uq_train_target_vals:
                        raise ValueError('Not include data of target value:' + str(uq_target_val) + ' in CVData.')


                self._model.fit(train_df.iloc[train_index, :], target_col[train_index], **cv_model_fit_params)

                tmp_train_predict_df = \
                    pd.DataFrame(self._model.predict_proba(train_df.iloc[test_index, :]), index=test_index,
                                 columns=[feature_name + '_is_' + str(x) for x in model.classes_])

                if train_predict_df is None:
                    train_predict_df = tmp_train_predict_df
                else:
                    train_predict_df = pd.concat([train_predict_df, tmp_train_predict_df], axis=0)

            self._model.fit(train_df, target_col, **model_fit_params)

            return train_predict_df

        else:
            raise ValueError('dtype of target_col is num or category')

    def apply(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to test_col by cached predict model.

        :param test_df: test dataframe to use converting ensemble features
        :return: Ensemble features of test_df

        :raise SyntaxError if not call train before
        """
        if self._feature_name is None:
            raise SyntaxError('apply function can use after calling fit function!')

        if dtype_is_num(self._target_dtype):
            # case: target dtype is num
            test_predicts: pd.Series = pd.Series(self._model.predict(test_df))
            return test_predicts.to_frame(self._feature_name)

        elif dtype_is_category(self._target_dtype):
            # case: target dtype is category
            return pd.DataFrame(self._model.predict_proba(test_df),
                                columns=[self._feature_name + '_is_' + str(x) for x in self._model.classes_])
        else:
            # never happen
            raise ValueError('dtype of target_col is num or category')
