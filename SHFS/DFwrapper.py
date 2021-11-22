import phik
import pandas as pd
import numpy as np 

class DFwrapper():
  """
  The Class helps to get rid of high correlation and outliers.
  Designed to work with Pandas DataFrame / Series
   

    Parametrs
    ----------
    thresh: float, default = None Set the threshold for the level of excessive correlation
    low: float, default = None Set the lower quantile to remove outliers
    high: float, default = None Set high quantile to remove outliers
    cols_clean: List, default = None List of attributes of the original set that need cleaning
    
    Methods
    ----------
    wrap_corr - removes unnecessary correlation, based on phik. Thresh parameter
    quantile_cleaner - Coarsely cuts high and low quantile. Low and high parameter
    frame_irq - finer cleaning of outliers. You need to pass koeff to calculate the swing.
                Recommended - 1.5
    get_bad - Returns a list of features with collinearity

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

   
    """
  def __init__(self, thresh=None, low=None, high=None, koeff=None):
    self.thresh = thresh
    self.koeff = koeff
    self.low = low
    self.high = high

  def wrap_corr(self, data):
    X = data.copy()
    phik_overview = X.phik_matrix().abs()
    upper_tri = phik_overview.where(np.triu(np.ones(phik_overview.shape), k=1).astype(np.bool))
    self.to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.thresh)]
    cleaned = X.drop(self.to_drop, axis=1)
    return cleaned

  def quantile_cleaner(self, data, cols_clean=None):

    df = data.copy()
    quant_df = df[cols_clean].quantile([self.low, self.high])
    df_clean = df[cols_clean].apply(lambda x: x[(x > quant_df.loc[self.low, x.name]) & 
                                     (x < quant_df.loc[self.high, x.name])], axis=0)
    df_clean.dropna(inplace=True)
    return df_clean

  def frame_irq(self, data, cols_clean=None, query=False):
    
    df = data.copy()
    
    if len(cols_clean)==0:
        cols_clean = list(df.columns)
    for column in cols_clean:
        q25 = df[column].quantile(.25)                 
        q75 = df[column].quantile(.75) 
        minimum = df[column].min()
        maximum = df[column].max()    
        irq = np.abs(q75 - q25)    
        left = q25 - self.koeff * irq
        right = q75 + self.koeff * irq
        if query: 
            sql_sentence='@left <= '+ column + ' and '+ column + ' <= @right'
            frames = df.query(sql_sentence)
        else:
            df_tmp = df[(df[column] > left ) & (df[column] < right)]
            if len(df_tmp) != 0:
                frames = df_tmp.copy()
        pass

    df_irq = pd.concat([frames])
    return df_irq

  def get_bad(self):
    return self.to_drop

