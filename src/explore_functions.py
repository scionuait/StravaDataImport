import pandas as pd
import numpy as np


def df_value_stat(df: pd.DataFrame) -> pd.DataFrame:
    """return dataframe with following data:
        columns_name
        value_range
        value_range_number
        dType
        null_value
        null_value_%

    Parameters
    ----------
   df: pd.DataFrame
        dataset
  

    Returns
    -------
    Dataset

    """
    df_tot_size = df.shape[0]
    list_to_export = []
    for c in df.columns:
        dict_to_insert = {"columns_name":c,\
                          "value_range":df[c].value_counts(),\
                          "value_range_number":np.size(df[c].value_counts()),\
                          "dType":df[c].dtypes,\
                          "null_value":df[c].isnull().sum(),\
                          "null_value_%":np.round(df[c].isnull().sum()/df_tot_size*100, decimals=2)}
        list_to_export.append(dict_to_insert)   
    
    return pd.DataFrame(list_to_export)