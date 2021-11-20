
 
<p align="center"><img src="https://i.ibb.co/ZXSk6jG/machine-learning-1920x1180.jpg" alt="machine-learning-1920x1180"></p>


 Класс вычисляет  важность признаков на основе библиотеки `Shap` для задачи классификации. 
  Работает только с деревьями для большей эффективности или моделями на основе 
  градиентного бустинга. Приоритетно использовать такие модели как:
   
   Catboost - не требует обработки `NaN` и категорий. работает с `sklearn`

  Для работы необходимо импортировать:

    from sklearn.base import BaseEstimator, TransformerMixin
    import shap

  Параметры. 
___
    `estimator` :   
        Обучение с учителем с методом fit позволит получить и отобрать индексы.  
        самых важных признаков. 
    n_features_to_select : int, default=None. 
        Количество признаков для отбора, по умолчанию значение None. 
    columns: List, default=None. 
        Список признаков исходного сета, по умолчанию значение None. 
    
  Методы
___
    fit - обучается и выявляет наиболее важные признаки
    tranform - изменяет исходный сет и вовзращает отобранные признаки
    get_index - возвращает отобранные признаки индексов
    plot_values - построение графика shap values
    _estimator_type - метод @property - возвращает тип модели
    get_feature_importance - возвращает DataFrame FI
  Примечание
___
    Не работает с категориальными признаками. Nan/Inf разрешены в случае, если 
    их принимает модель метода fit
  Пример использования для классификации
___
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
       
Пример без pipline

       cols = list(X_train.columns)
       estimator = CatBoostClassifier(**params_cat)
       selector = ShapFeatureSelection(estimator, n_features_to_select=3, columns=cols)
       X = selector.fit(X_train_prep, y_train)

