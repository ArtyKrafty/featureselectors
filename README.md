<p align="center"><img src="https://i.ibb.co/ZXSk6jG/machine-learning-1920x1180.jpg" alt="machine-learning-1920x1180"></p>
 
Library consist of two groups of Classes - `Featureselectors` and `DFwrapper` to have a deal with outliers and correlation

1. Feature selection group

The FeatureSelection calculates the importance of features based on the `Shap` library for a classification problem.
  Only works with trees for better efficiency or models based on
  gradient boosting. It is a priority to use such models as:
   
   Catboost - does not require handling of `NaN` and categories. works with `sklearn`

    NOTE: If your import is failing due to a missing package, you can
    manually install dependencies using either !pip or !apt.

            !pip install shap 
            !pip install phik
   
  https://pypi.org/project/SHFS/
  

            FeatureSelectionClf - for classification
            FeatureSelectionRegression - for regression
            FeatureSelectionUniversal - for both classification and regression tasks

  Quick start: [Collab](https://colab.research.google.com/drive/1eP6qZmxcTcsKgjLL7u_pHaM5sZc8346N?usp=sharing) and [Tutorial](https://nbviewer.org/github/ArtyKrafty/featureselectors/blob/main/Tutorial/Tutorials_ipynb_.ipynb)
        

  Parametrs. 
___
    `estimator` :   
        Supervised learning with the fit method will allow you to retrieve and select indices.
        the most important features.
    n_features_to_select: int, default = None.
        The number of features to select, the default is None.
    columns: List, default = None.
        The list of attributes of the initial set, the default is None.
    
  Methods
___
    fit - trains and identifies the most important features
    tranform - changes the original set and returns the selected attributes
    get_index - Returns the selected indexes attributes

    only for FeatureSelectionClf and FeatureSelectionRegression:

    plot_values - plotting shap values
    _estimator_type - @property method 
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


2. DFwrapper

DFwrapper - remove multicollinearity and outliers from Pandas DataFrame

    Usage example
    ----------
    1. Collinearity

    cleaner = DFwrapper()
    new_df = cleaner.wrap_corr(df)

    2. Outliers. Rough cleaning

    cleaner = DFwrapper(low=.05, high=.95)
    cleaned = cleaner.quantile_cleaner(df, cols_to_clean)

    2. Outliers. Finer cleaning

    cleaner = DFwrapper(koeff=1.5)
    cleaned = cleaner.frame_irq(df, cols_to_clean)

