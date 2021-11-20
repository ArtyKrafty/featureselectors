<p align="center"><img src="https://i.ibb.co/ZXSk6jG/machine-learning-1920x1180.jpg" alt="machine-learning-1920x1180"></p>
 



The class calculates the importance of features based on the `Shap` library for a classification problem.
  Only works with trees for better efficiency or models based on
  gradient boosting. It is a priority to use such models as:
   
   Catboost - does not require handling of `NaN` and categories. works with `sklearn`
   
  https://pypi.org/project/SHFS/
  
  you need to import:
  
  Quick start: [Collab](https://colab.research.google.com/gist/ArtyKrafty/5a6cb7ab1bf9366e4f93f44f316549b1/example.ipynb)

    !pip install shap 
    !pip install catboost 
    !pip install SHFS 

    from sklearn.base import BaseEstimator, TransformerMixin
    import shap



  Parametrs. 
___
    `estimator` :   
        Supervised learning with the fit method will allow you to retrieve and select indices.
        the most important signs.
    n_features_to_select: int, default = None.
        The number of features to select, the default is None.
    columns: List, default = None.
        The list of attributes of the initial set, the default is None.
    
  Methods
___
    fit - trains and identifies the most important signs
    tranform - changes the original set and returns the selected attributes
    get_index - Returns the selected indexes attributes
    plot_values - plotting shap values
    _estimator_type - @property method - returns the type of the model
    get_feature_importance - Returns DataFrame FI
  Note
___
 Nan / Inf are allowed in case
    they are accepted by the fit method model
  Example use for classification
___
    cols = list(X_train.columns)
    cat_features = list(X_train_cat.select_dtypes(include=['object', 'category']).columns)
    num_features = list(X_train_cat.select_dtypes(exclude=['object', 'category']).columns)
    estimator = CatBoostClassifier(**params_cat)
    selector = FeatureSelectionClf(estimator, n_features_to_select=3, columns=cols) 
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
       
Example without Pipeline

       cols = list(X_train.columns)
       estimator = CatBoostClassifier(**params_cat)
       selector = FeatureSelectionClf(estimator, n_features_to_select=3, columns=cols)
       X = selector.fit(X_train_prep, y_train)

