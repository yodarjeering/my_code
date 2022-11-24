import pandas as pd
import numpy as np

class MakeTrainData():
    

    def __init__(self, df_con, test_rate=0.9, is_bit_search=False,is_category=True,ma_short=5,ma_long=25):
        self.df_con = df_con
        self.test_rate = test_rate
        self.is_bit_search = is_bit_search
        self.is_category = is_category
        self.ma_short = ma_short
        self.ma_long = ma_long

                
    def add_ma(self):
        df_process = self.df_con.copy()
        df_process['ma_short'] = df_process['close'].rolling(self.ma_short).mean()
        df_process['ma_long']  = df_process['close'].rolling(self.ma_long).mean()
        df_process['std_short'] = df_process['close'].rolling(self.ma_short).std()
        df_process['std_long']  = df_process['close'].rolling(self.ma_long).std()
        df_process['ema_short'] = df_process['close'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['ema_long'] = df_process['close'].ewm(span=self.ma_long, adjust=False).mean()
        df_process['macd'] = df_process['ema_short'] - df_process['ema_long']
        df_process['macd_signal_short'] = df_process['macd'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['macd_signal_long'] = df_process['macd'].ewm(span=self.ma_long, adjust=False).mean()
        return df_process
                
        
    def make_data(self,is_check=False):
        x = pd.DataFrame(index=self.df_con.index)
        # この書き方は環境によってはエラー
        # x.index = self.df_con.index
        df_con = self.df_con.copy()
        df_ma = self.add_ma()
        end_point = -1
        if is_check:
            end_point = len(self.df_con)
        else:
            end_point = len(self.df_con)-1
        
        # ダウ変化率
        dawp_5 = df_con['dclose'].iloc[:-5]
        dawp_5.index = df_con.index[5:]
        x['dawp_5'] = dawp_5
        dawp_0 = df_con['dclose']
        x['dawp_0'] = dawp_0
        
        # 日経変化率
        nikkeip_5 = df_con['pclose'].iloc[:-5]
        nikkeip_5.index = df_con.index[5:]
        x['nikkeip_5'] = nikkeip_5
        nikkeip_0 = df_con['pclose']
        x['nikkeip_0'] = nikkeip_0
        
        # high - low 変化率
        high_low = (df_con['high']-df_con['low'])/df_con['close']
        x['diff_rate'] = high_low
        
        # close - open 変化率
        close_open = (df_con['close']-df_con['open'])/df_con['close']
        x['close_open'] = close_open
        
        # 売買量変化率
        nikkei_volumep = df_con['volume'].pct_change()
        x['nikkei_volumep'] = nikkei_volumep
        
        # 短期標準偏差ベクトル
        std_s_5 = df_ma['std_short'].iloc[:-5]
        std_s_5.index = df_ma.index[5:]
        x['std_s_5'] = std_s_5
        std_s_0 = df_ma['std_short']
        x['std_s_0'] = std_s_0
        
        # 長期標準偏差ベクトル
        std_l_5 = df_ma['std_long'].iloc[:-5]
        std_l_5.index = df_ma.index[5:]
        x['std_l_5'] = std_l_5
        std_l_0 = df_ma['std_long']
        x['std_l_0'] = std_l_0
        
        # 短期移動平均ベクトル
        vec_s_5 = (df_ma['ma_short'].diff(5)/5)
        x['vec_s_5'] = vec_s_5
        vec_s_1 = (df_ma['ma_short'].diff(1)/1)
        x['vec_s_1'] = vec_s_1
        
        # 長期移動平均ベクトル
        vec_l_5 = (df_ma['ma_long'].diff(5)/5)
        x['vec_l_5'] = vec_l_5
        vec_l_1 = (df_ma['ma_long'].diff(1)/1)
        x['vec_l_1'] = vec_l_1
        
#         移動平均乖離率
        x['d_MASL'] = df_ma['ma_short']/df_ma['ma_long']

#          ema短期のベクトル
        emavec_s_5 = (df_ma['ema_short'].diff(5)/5)
        x['emavec_s_5'] = emavec_s_5
        emavec_s_1 = (df_ma['ema_short'].diff(1)/1)
        emavec_s_1.index = df_ma.index
        x['emavec_s_1'] = emavec_s_1
    
        # ema長期ベクトル
        emavec_l_5 = (df_ma['ema_long'].diff(5)/5)
        x['emavec_l_5'] = emavec_l_5
        emavec_l_1 = (df_ma['ema_long'].diff(1)/1)
        x['emavec_l_1'] = emavec_l_1

        #         EMA移動平均乖離率
        x['d_EMASL'] = df_ma['ema_short']/df_ma['ema_long']
        
        # macd
        macd = df_ma['macd']
        x['macd'] = macd
        macd_signal_short = df_ma['macd_signal_short']
        x['macd_signal_short'] = macd_signal_short
        macd_signal_long = df_ma['macd_signal_long']
        x['macd_signal_long'] = macd_signal_long
            
        # 短期相関係数
        df_tmp1 = df_con[['close','daw_close']].rolling(self.ma_short).corr()
        corr_short = df_tmp1.drop(df_tmp1.index[0:-1:2])['close']
        corr_short = corr_short.reset_index().set_index('day')['close']
        x['corr_short'] = corr_short
        
        # 長期相関係数
        df_tmp2 = df_con[['close','daw_close']].rolling(self.ma_long).corr()
        corr_long = df_tmp2.drop(df_tmp2.index[0:-1:2])['close']
        corr_long = corr_long.reset_index().set_index('day')['close']
        x['corr_long'] = corr_long
        
        # 歪度
        skew_short = df_con['close'].rolling(self.ma_short).skew()
        x['skew_short'] = skew_short
        skew_long = df_con['close'].rolling(self.ma_long).skew()
        x['skew_long'] = skew_long
        
        # 尖度
        kurt_short = df_con['close'].rolling(self.ma_short).kurt()
        x['kurt_short'] = kurt_short
        kurt_long = df_con['close'].rolling(self.ma_long).kurt()
        x['kurt_long'] = kurt_long
        
        # RSI 相対力指数
        df_up = df_con['dclose'].copy()
        df_down = df_con['dclose'].copy()
        df_up[df_up<0] = 0
        df_down[df_down>0] = 0
        df_down *= -1
        sims_up = df_up.rolling(self.ma_short).mean()
        sims_down = df_down.rolling(self.ma_short).mean()
        siml_up = df_up.rolling(self.ma_long).mean()
        siml_down = df_down.rolling(self.ma_long).mean()
        RSI_short = sims_up / (sims_up + sims_down) * 100
        RSI_long = siml_up / (siml_up + siml_down) * 100
        x['RSI_short'] = RSI_short
        x['RSI_long'] = RSI_long
        
        
        open_ =  df_con['open']
        high_ = df_con['high']
        low_ = df_con['low']
        close_ = df_con['close']

#        Open Close 乖離率
        x['d_OC'] = open_/close_

#       High low 乖離率
        x['d_HL'] = high_/low_
        df_atr = pd.DataFrame(index=high_.index)
        df_atr['high_low'] = high_ - low_
        df_atr['high_close'] = high_ - close_
        df_atr['close_low_abs'] =  (close_ - low_).abs()
        tr = pd.DataFrame(index=open_.index)
        tr['TR'] = df_atr.max(axis=1)

        # ATR
        x['ATR_short'] = tr['TR'].rolling(self.ma_short).mean()
        x['ATR_long'] =  tr['TR'].rolling(self.ma_long).mean()
        x['d_ATR'] = x['ATR_short']/x['ATR_long']
        x['ATR_vecs5'] = (x['ATR_short'].diff(5)/1)
        x['ATR_vecs1'] = (x['ATR_short'].diff(1)/1)
        x['ATR_vecl5'] = (x['ATR_long'].diff(5)/1)
        x['ATR_vecl1'] = (x['ATR_long'].diff(1)/1)
        
        today_close = df_con['close']
        yesterday_close = df_con['close'].iloc[:-1]
        yesterday_close.index = df_con.index[1:]
#        騰落率
#       一度も使用されていなかったため, 削除
        # x['RAF'] =  (today_close/yesterday_close -1)
        x = x.iloc[self.ma_long:end_point]
        x_check = x
#        この '4' は　std_l5 など, インデックスをずらす特徴量が, nanになってしまう分の日数を除くためのもの
        # yについても同様
        x_train = x.iloc[self.ma_short-1:int(len(x)*self.test_rate)]
        x_test  = x.iloc[int(len(x)*self.test_rate):]


        if not is_check:
            
            y_train,y_test = self.make_y_data(x,self.df_con,end_point)
            return x_train, y_train, x_test, y_test
        
        
        else:
            x_check = x_check.iloc[self.ma_short-1:]
            chart_ = self.df_con.loc[x_check.index]
            
            return x_check,chart_


    def make_y_data(self,x,df_con,end_point):
        y = []
        for i in range(self.ma_long,end_point):
            tommorow_close = df_con['close'].iloc[i+1]
            today_close    = df_con['close'].iloc[i]
            if tommorow_close>today_close:
                y.append(1)
            else:
                y.append(0)
            
        y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
        y_test  = y[int(len(x)*self.test_rate):]
        return y_train,y_test

class MakeTrainData3(MakeTrainData):

# alpha=0.55 ?
    def __init__(self,df_con,test_rate=0.8,alpha=0.5,beta=-0.4):
        super(MakeTrainData3,self).__init__(df_con,test_rate=test_rate)
        self.alpha = alpha
        self.beta = beta


    def make_y_data(self,x,df_con,end_point):
        y = []
        for i in range(self.ma_long,end_point):
            tommorow_close = df_con['close'].iloc[i+1]
            today_close    = df_con['close'].iloc[i]
            change_rate = ((tommorow_close-today_close) / today_close) * 100
            
            # UP : 2
            if change_rate > self.alpha:
                y.append(2)
            # DOWN : 0
            elif change_rate < self.beta:
                y.append(0)
            # STAY : 1
            else:
                y.append(1)
            
        y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
        y_test  = y[int(len(x)*self.test_rate):]
        return y_train,y_test
