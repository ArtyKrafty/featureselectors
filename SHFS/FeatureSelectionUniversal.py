import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelectionUniversal(BaseEstimator, TransformerMixin) :
    """
    The class calculates the importance of features based on the Shap library for a regression problem
    Only works with trees for better efficiency or models based on
    gradient boosting. It is a priority to use such models as:

    Catboost - does not require NaN and category handling. works with sklearn

    To work, you need to import:

      from sklearn.base import BaseEstimator, TransformerMixin
      import shap

    Options
      ----------
      estimator:
          Supervised learning with fit method will get and select indices
          the most important signs
      n_features_to_select: int, default = None
          Number of features for selection, default value is None
      columns: List, default = None
          List of attributes of the initial set, default value is None

 
    Methods
      ----------
      fit - trains and identifies the most important signs
      tranform - changes the original set and returns the selected attributes
      get_index - Returns the selected indexes attributes

    Note
      -----
      Categorical signs and Nan / Inf are allowed if
      they are accepted by the fit method model
    Usage example
      ----------
      cols = list (X_train.columns)
      estimator = CatBoostRegressor (** params)
      selector = FeatureSelectionUniversal (estimator, n_features_to_select = 20, columns = cols)
      selector.fit (X_train, y_train)


      """

    def __init__(self, estimator, n_features_to_select = None, columns = None) :
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.columns = columns

    def fit(self, X, y, **fit_params) :

        X_train, self.X_val, y_train, y_val = train_test_split(
            X, y, test_size = .25, random_state = 42
            )

        assert X_train.shape[0] + self.X_val.shape[0] == X.shape[0], 'Неправильно разделена выборка'

        self.estimator.fit(
            X_train, y_train, eval_set = [(self.X_val, y_val)],
            early_stopping_rounds = 200
            , use_best_model = True
            )

        self.fi = (pd.DataFrame(
            {'feature' : self.columns,
             'importance' : self.estimator.feature_importances_}
            )
                   .sort_values(by = 'importance', ascending = False))
        self.fi.sort_values(by = ['importance'], ascending = False, inplace = True)
        self.idx = list(self.fi['importance'].index)[:self.n_features_to_select]

        return self

    def transform(self, X, y = None) :
        try :
            X_selected = X[:, self.idx]
        except :
            X_selected = X.iloc[:, self.idx]
        return X_selected

    def get_index(self) :
        return self.idx
