import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelectionUniversal(BaseEstimator, TransformerMixin) :
    """
    Класс вычисляет  важность признаков на основе библиотеки Shap для задачи регрессии
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

    Примечание
      -----
      Категориальные признаки и Nan/Inf разрешены в случае, если
      их принимает модель метода fit
    Пример использования
      ----------
      cols = list(X_train.columns)
      estimator = CatBoostRegressor(**params)
      selector = FeatureSelection(estimator, n_features_to_select=20, columns=cols)
      selector.fit(X_train, y_train)


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
