import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelectionRegression(BaseEstimator, TransformerMixin) :
    """
    The class calculates the importance of features based on the Shap library for a regression problem.
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
      plot_values - plotting shap values
      _estimator_type - @property method - returns the type of the model
      get_feature_importance - Returns DataFrame FI
    Note
      -----

      ----------
      cols = list(X_train.columns)
      cat_features = list(X_train_cat.select_dtypes(include=['object', 'category']).columns)
      num_features = list(X_train_cat.select_dtypes(exclude=['object', 'category']).columns)
      estimator = CatBoostRegressor(**params_cat)
      selector = ShapFeatureSelection(estimator, n_features_to_select=3, columns=cols)


      preprocessor = ColumnTransformer (
          transformers = [

              ('std_scaler' , StandardScaler() , num_features) ,
              ('cat' , OrdinalEncoder() , cat_features),


              ]
      )

      pipe = Pipeline(steps=

          [
            ('preprocessor', preprocessor),
            ('selector', selector)

          ]
      )

     X_train_prep = pipe.fit_transform(X_train)


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

        preds = self.estimator.predict(self.X_val)
        explainer = shap.TreeExplainer(self.estimator)
        self.shap_values = explainer.shap_values(self.X_val)

        try :

            vals = np.abs(self.shap_values).mean(0)
            self.feature_importance = pd.DataFrame(
                list(zip(self.columns, vals)),
                columns = ['col_name', 'feature_importance_vals']
                )
            self.feature_importance.sort_values(
                by = ['feature_importance_vals'],
                ascending = False, inplace = True
                )
            self.idx = (list(self.feature_importance['feature_importance_vals'].index)
            [:self.n_features_to_select]
            )

        except :

            vals = np.abs(self.shap_values[:1]).mean(0)
            self.feature_importance = pd.DataFrame(
                list(zip(self.columns, vals)),
                columns = ['col_name', 'feature_importance_vals']
                )
            self.feature_importance.sort_values(
                by = ['feature_importance_vals'],
                ascending = False, inplace = True
                )
            self.idx = (list(self.feature_importance['feature_importance_vals'].index)
            [:self.n_features_to_select]
            )

        return self

    def transform(self, X, y = None) :
        try :
            X_selected = X[:, self.idx]
        except :
            X_selected = X.iloc[:, self.idx]
        return X_selected

    def get_index(self) :
        return self.idx

    def plot_values(self) :
        return shap.summary_plot(self.shap_values, self.X_val)

    @property
    def _estimator_type(self) :
        return self.estimator._estimator_type

    def get_feature_importance(self) :
        return self.feature_importance
