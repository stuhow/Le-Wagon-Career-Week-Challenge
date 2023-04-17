# preprocessing
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class custom_imputer(TransformerMixin, BaseEstimator):
    # BaseEstimator generates the get_params() and set_params() methods that all Pipelines require
    # TransformerMixin creates the fit_transform() method from fit() and transform()

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # stateless transormation so no need to fit
        return self

    def transform(self, X, y=None):

        X= X.fillna(0)

        return X

class time_tranformer(TransformerMixin, BaseEstimator):
    # BaseEstimator generates the get_params() and set_params() methods that all Pipelines require
    # TransformerMixin creates the fit_transform() method from fit() and transform()

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # stateless transormation so no need to fit
        return self

    def transform(self, X, y=None):

        hours_in_a_day = 24

        X['sin_time_hours'] = np.sin(2*np.pi*(X.time_hours)/hours_in_a_day)
        X['cos_time_hours'] = np.cos(2*np.pi*(X.time_hours)/hours_in_a_day)
        X.drop(columns=['time_hours'], inplace=True)

        return X
