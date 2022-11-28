import pandas as pd
import glob
import numpy as np

def process_kawase(path_kawase,df_con):
    df_kawase = pd.read_csv(path_kawase, index_col=0,encoding='Shift_JIS')
    column_name = df_kawase.iloc[0]
    df_kawase = df_kawase.set_axis(df_kawase.iloc[1].values.tolist(),axis=1).iloc[2:]
    df_kawase.dropna(how='all',axis=1,inplace=True)
    df_kawase.replace('*****',np.nan,inplace=True)
    df_kawase = df_kawase.astype('float64')
    df_kawase['day'] = pd.to_datetime(df_kawase.index,format='%Y/%m/%d')
    df_kawase.set_index('day',inplace=True)
    return df_kawase

class DataFramePreProcessing():

    
    def __init__(self, path_, is_daw=False):
        self.path_ = path_
        self.is_daw = is_daw

        
    def load_df(self):
        if self.is_daw:
            d='d'
        else:
            d=''
        FILE = glob.glob(self.path_)
        df = pd.read_csv(FILE[0])
        df = df.rename(columns={df.columns[0]:'nan',df.columns[1]:'nan',df.columns[2]:'nan',\
                                    df.columns[3]:'day',df.columns[4]:'nan',df.columns[5]:d+'open',\
                                    df.columns[6]:d+'high',df.columns[7]:d+'low',df.columns[8]:d+'close',\
                df.columns[9]:d+'volume',})
        df = df.drop('nan',axis=1)
        df = df.drop(df.index[0])
        df['day'] = pd.to_datetime(df['day'],format='%Y/%m/%d')
        df.set_index('day',inplace=True)

        return df.astype(float)
    
