import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin


class FeatureSelectionClf(BaseEstimator,TransformerMixin):
    """
    Класс вычисляет  важность признаков на основе библиотеки Shap для задачи классификации.
    Работает только с деревьями для большей эффективности или моделями на основе
    градиентного бустинга. Приоритетно использовать такие модели как:

     Catboost - не требует обработки NaN и категорий. работает с sklearn

    Для работы необходимо импортировать:

      from sklearn.base import BaseEstimator, TransformerMixin
      import shap

    Параметры
      ----------
      estimator :
          Обучение с учителем с методом fit позволит получить и отобрать индексы
          самых важных признаков
      n_features_to_select : int, default=None
          Количество признаков для отбора, по умолчанию значение None
      columns: List, default=None
          Список признаков исходного сета, по умолчанию значение None

    Методы
      ----------
      fit - обучается и выявляет наиболее важные признаки
      tranform - изменяет исходный сет и вовзращает отобранные признаки
      get_index - возвращает отобранные признаки индексов
      plot_values - построение графика shap values
      _estimator_type - метод @property - возвращает тип модели
      get_feature_importance - возвращает DataFrame FI
    Примечание
      -----
      Не работает с категориальными признаками. Nan/Inf разрешены в случае, если
      их принимает модель метода fit
    Пример использования
      ----------
      cols = list(X_train.columns)
      cat_features = list(X_train_cat.select_dtypes(include=['object', 'category']).columns)
      num_features = list(X_train_cat.select_dtypes(exclude=['object', 'category']).columns)
      estimator = CatBoostClassifier(**params_cat)
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

    def __init__(self,estimator,n_features_to_select=None,columns=None):
        self.estimator=estimator
        self.n_features_to_select=n_features_to_select
        self.columns=columns

    def fit(self,X,y,**fit_params):

        X_train,self.X_val,y_train,y_val=train_test_split(
            X,y,test_size=.25,random_state=42
            )

        assert X_train.shape[0]+self.X_val.shape[0]==X.shape[0],'Неправильно разделена выборка'

        self.estimator.fit(X_train,y_train)

        preds=self.estimator.predict_proba(self.X_val)[:,1]
        explainer=shap.TreeExplainer(self.estimator)
        self.shap_values=explainer.shap_values(self.X_val)

        try:

            vals=np.abs(self.shap_values).mean(0)
            self.feature_importance=pd.DataFrame(
                list(zip(self.columns,vals)),
                columns=['col_name','feature_importance_vals']
                )
            self.feature_importance.sort_values(
                by=['feature_importance_vals'],
                ascending=False,inplace=True
                )
            self.idx=(list(self.feature_importance['feature_importance_vals'].index)
            [:self.n_features_to_select]
            )

        except:

            vals=np.abs(self.shap_values[:1]).mean(0)
            self.feature_importance=pd.DataFrame(
                list(zip(self.columns,vals)),
                columns=['col_name','feature_importance_vals']
                )
            self.feature_importance.sort_values(
                by=['feature_importance_vals'],
                ascending=False,inplace=True
                )
            self.idx=(list(self.feature_importance['feature_importance_vals'].index)
            [:self.n_features_to_select]
            )

        return self

    def transform(self,X,y=None):
        try:
            X_selected=X[:,self.idx]
        except:
            X_selected=X.iloc[:,self.idx]
        return X_selected

    def get_index(self):
        return self.idx

    def plot_values(self):
        return shap.summary_plot(self.shap_values,self.X_val)

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def get_feature_importance(self):
        return self.feature_importance
