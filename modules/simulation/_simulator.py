import pandas as pd
import numpy as np
from scipy import signal,fftpack
import matplotlib.pyplot as plt
import random
from modules.preprocessing import MakeTrainData,MakeTrainData3
from modules.preprocessing import DataFramePreProcessing
from modules.preprocessing import PlotTrade
from modules.funcs import cos_sim,standarize,norm,butter_lowpass_filter
from modules.training import LearnClustering


class Simulation():


    def __init__(self):
        self.model = None
        self.accuracy_df = None
        self.trade_log = None
        self.pr_log = None
        self.MK = MakeTrainData
        self.ma_short =  5
        self.ma_long = 25
        self.wallet = 2500


    def simulate_routine(self, path_tpx, path_daw,start_year=2021,end_year=2021,start_month=1,end_month=12,df_="None",is_validate=False):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        df_con = self.make_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
    
        # 任意の期間の df を入力しても対応できる
        if type(df_)==pd.DataFrame or type(df_)==pd.Series:
            start_ = df_.index[0]
            end_ = df_.index[-1]
            x_check = x_check.loc[start_:end_]
            y_ = y_.loc[start_:end_]
            df_con = df_con.loc[start_:end_]
        else:
            x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
            y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
            df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)

        self.df_con = df_con
        y_check = y_.values.reshape(-1).tolist()
        if not is_validate:
            pl = PlotTrade(df_con['close'],label='終値')
            pl.add_plot(df_con['ma_short'],label='短期移動平均線')
            pl.add_plot(df_con['ma_long'],label='長期移動平均線')
        else:
            pl=None
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()

        return x_check,y_check,y_,df_con,pl


    def set_for_online(self,x_check,y_):
        x_tmp = x_check
        y_tmp = y_
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame(index=x_tmp.index)
        acc_df['pred'] = [-1] * len(acc_df)
        return x_tmp, y_tmp, current_date, acc_df


        
    def learn_online(self,x_tmp,y_tmp,x_check,current_date,tmp_date):
        x_ = x_tmp[current_date<=x_tmp.index]
        x_ = x_[x_.index<tmp_date]
        y_ = y_tmp[current_date<=y_tmp.index]
        y_ = y_[y_.index<tmp_date]
        self.xgb_model = self.xgb_model.fit(x_,y_)
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        current_date = tmp_date

        return predict_proba, current_date


    def buy(self,df_con,x_check,i):
#   観測した始値が, 予測に反して上がっていた時, 買わない
        index_buy = df_con['close'].loc[x_check.index[i+1]]
        start_time = x_check.index[i+1]
        is_bought = True

        return index_buy, start_time, is_bought


    def sell(self,df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate):
        index_sell = df_con['close'].loc[x_check.index[i+1]]
        end_time = x_check.index[i+1]
        prf += index_sell - index_buy
        prf_list.append(index_sell - index_buy)
        trade_count += 1
        is_bought = False
        if not is_validate:
            pl.add_span(start_time,end_time)
        else:
            pass

        return prf, trade_count, is_bought


    def hold(self,df_con,index_buy,total_eval_price,i):
        eval_price = df_con['close'].iloc[i] - index_buy
        total_eval_price += eval_price
        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
        return total_eval_price


# SELL の直後に BUY となる時, シミュレートできてない
    def return_grad(self, df, index, gamma=0, delta=0):
        grad_ma_short = df['ma_short'].iloc[index+1] - df['ma_short'].iloc[index]
        grad_ma_long  = df['ma_long'].iloc[index+1] - df['ma_long'].iloc[index]
        strategy = ''
        
        if grad_ma_long >= gamma:
            strategy = 'normal'
        elif grad_ma_long < delta:
            strategy = 'reverse'
        else:
            print("No such threshold")
        return strategy

    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose'],tpx_p['pclose']],axis = 1,join='inner').astype(float)
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con

    
    def make_check_data(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=1.0)
        x_check, y_check, _, _ = mk.make_data()
        # self.ma_short = mk.ma_short
        # self.ma_long = mk.ma_long
        return x_check, y_check
    

    def calc_acc(self, acc_df, y_check):
        df = pd.DataFrame(columns = ['score','Up precision','Down precision','Up recall','Down recall','up_num','down_num'])
        acc_dict = {'TU':0,'FU':0,'TD':0,'FD':0}
    
        for i in range(len(acc_df)):
            
            label = acc_df['pred'].iloc[i]
            if label==-1:continue

            if y_check[i]==label:
                if label==0:
                    acc_dict['TD'] += 1
                else:#label = 1 : UP
                    acc_dict['TU'] += 1
            else:
                if label==0:
                    acc_dict['FD'] += 1
                else:
                    acc_dict['FU'] += 1

        df = self.calc_accuracy(acc_dict,df)
        return df


    def calc_accuracy(self,acc_dict,df):
        denom = 0
        for idx, key in enumerate(acc_dict):
            denom += acc_dict[key]
        
        try:
            TU = acc_dict['TU']
            FU = acc_dict['FU']
            TD = acc_dict['TD']
            FD = acc_dict['FD']
            score = (TU + TD)/(denom)
            prec_u = TU/(TU + FU)
            prec_d = TD/(TD + FD)
            recall_u = TU/(TU + FD)
            recall_d = TD/(TD + FU)
            up_num = TU+FD
            down_num = TD+FU
            col_list = [score,prec_u,prec_d,recall_u,recall_d,up_num,down_num]
            df.loc[0] = col_list
            return df
        except:
            print("division by zero")
            return None


# ここ間違ってる
    def return_split_df(self,df,start_year=2021,end_year=2021,start_month=1,end_month=12):
        df = df[df.index.year>=start_year]
        if start_year <= end_year:
            df = df[df.index.year<=end_year]
        if len(set(df.index.year))==1:
            df = df[df.index.month>=start_month]
            df = df[df.index.month<=end_month]
        else:
            df_tmp = df[df.index.year==start_year]
            last_year_index = df_tmp[df_tmp.index.month==start_month].index[0]
#             new_year_index = df[df.index.month==end_year].index[-1]
            df = df.loc[last_year_index:]
        return df


    def return_trade_log(self,prf,trade_count,prf_array,cant_buy):
        
        if trade_count==0:
            max_profit = 0
            min_profit = 0
            mean_profit = 0
        else:
            max_profit = prf_array.max()
            min_profit = prf_array.min()
            mean_profit= prf_array.mean()
            
        pr = (prf/self.wallet)*100
        log_dict = {
            'total_profit':prf,
            'profit rate':pr,
            'trade_count':trade_count,
            'max_profit':max_profit,
            'min_profit':min_profit,
            'mean_profit':mean_profit,
            'cant_buy_count':cant_buy
            }
        df = pd.DataFrame(log_dict,index=[1])
        return df


    
    def get_accuracy(self):
        return self.accuracy_df


    def get_trade_log(self):
        return self.trade_log


    def simulate(self):
        pass


# simulate 済みを仮定
    def return_profit_rate(self,wallet=2500):
        self.pr_log['reward'] = self.pr_log['reward'].map(lambda x: x/wallet)
        self.pr_log['eval_reward'] = self.pr_log['eval_reward'].map(lambda x: x/wallet)
        return self.pr_log

class XGBSimulation(Simulation):
    
    
    def __init__(self, lx, alpha=0.70):
        super().__init__()
        self.lx = lx
        self.xgb_model = lx.model
        self.alpha = alpha
        self.acc_df = None
        self.y_check = None
        self.ma_long = 25
        self.ma_short = 5
        self.is_bought = False
        # 20日以上 ホールドしたら, 自動的に売り
        self.calc_func = self.calc_acc
        self.MK = MakeTrainData
        

    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
                is_variable_strategy=False,is_observed=False,df_="None"):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,df_,is_validate)
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trade_count = 0
        trigger_count = 0
        total_eval_price = 0
        cant_buy = 0 # is_observed=True としたことで買えなくなった取引の回数をカウント
        is_trigger=False
        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
#             label==0 -> down
#             label==1 -> up
#             オンライン学習
            tmp_date = x_tmp.index[i]   
            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)
# ここのprob は2クラスうち, 出力の大きいほうのクラスの可能性が代入されている
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                else: # label == 1:  
                    acc_df.iloc[i] = 1
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i-1,gamma=0, delta=0)
            

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==1 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==1 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            else:
                print("No such strategy.")
                return 

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger=True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                print(log)
                print("")
                print(df)
                print("")
                print("trigger_count :",trigger_count)
                pl.show()
        except:
            print("no trade")
    
    def show_result(self, path_tpx,path_daw,strategy='normal'):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)  
        self.simulate(x_check,y_check,strategy)

class XGBSimulation2(XGBSimulation):


    def __init__(self,lx,alpha=0.33):
        super().__init__(lx,alpha)
        self.MK = MakeTrainData3

    def calc_acc(self, acc_df, y_check):
        df = pd.DataFrame(columns = ['score','Up precision','Stay precision','Down precision','Up recall','Stay recall','Down recall','up_num','stay_num','down_num'])
        acc_dict = {'TU':0,'FUs':0,'FUd':0,'TS':0,'FSu':0,'FSd':0,'TD':0,'FDu':0,'FDs':0}
    
        for i in range(len(acc_df)):
            
            label = acc_df['pred'].iloc[i]
            if label == -1: continue

            if y_check[i]==label:
                if label==0:
                    acc_dict['TD'] += 1
                elif label==1:
                    acc_dict['TS'] += 1
                else:#label = 2 : UP
                    acc_dict['TU'] += 1
            else:
                if label==0:
                    if y_check[i]==2:
                        # FDu
                        acc_dict['FDu'] += 1
                    else: 
                        # FDs
                        acc_dict['FDs'] += 1
                elif label==1:
                    if y_check[i]==2:
                        # FSu
                        acc_dict['FSu'] += 1
                    else:
                        # FSd
                        acc_dict['FSd'] += 1
                else:
                    if y_check[i]==0:
                        # FUd
                        acc_dict['FUd'] += 1
                    else:
                        # FUs
                        acc_dict['FUs'] += 1

        df = self.calc_accuracy(acc_dict,df)
        return df


    def calc_accuracy(self,acc_dict,df):
        denom = 0
        for idx, key in enumerate(acc_dict):
            denom += acc_dict[key]
        
        try:
            TU = acc_dict['TU']
            FUs = acc_dict['FUs']
            FUd = acc_dict['FUd']
            TS = acc_dict['TS']
            FSu = acc_dict['FSu']
            FSd = acc_dict['FSd']
            TD = acc_dict['TD']
            FDu = acc_dict['FDu']
            FDs = acc_dict['FDs']

            score = (TU + TD + TS)/(denom)
            prec_u = TU/(TU + FUs + FUd)
            prec_s = TS/(TS + FSu + FSd)
            prec_d = TD/(TD + FDu + FDs)
            recall_u = TU/(TU + FSu + FDu)
            recall_s = TS/(TS + FUs + FDs)
            recall_d = TD/(TD + FUd + FSd)
            # recall の分母
            up_num = TU+FSu+FDu
            stay_num = TS+FUs+FDs
            down_num = TD+FUd+FSd
            col_list = [score,prec_u,prec_s,prec_d,recall_u,recall_s,recall_d,up_num,stay_num,down_num]
            df.loc[0] = col_list
            return df
        except:
            print("division by zero")
            return None



    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
            is_variable_strategy=False,is_observed=False,df_="None"):
    
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,df_,is_validate)
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0 # is_observed=True としたことで買えなくなった取引の回数をカウント

        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)

# ここのprob は2クラスうち, 出力の大きいほうのクラスの可能性が代入されている
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i-1,gamma=0, delta=0)
            

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            else:
                print("No such strategy.")
                return 

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            pl.show()

class TechnicalSimulation(Simulation):
    
    
    def __init__(self,ma_short=5, ma_long=25, hold_day=5):
        super().__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.hold_day = hold_day
        
    
    def is_buyable(self, short_line, long_line, index_):
#         1=<index<=len-1 仮定
        long_is_upper = long_line.iloc[index_-1]>short_line.iloc[index_-1]
        long_is_lower = long_line.iloc[index_]<=short_line.iloc[index_]
        buyable = long_is_upper and long_is_lower
        return buyable
    
    
    def is_sellable(self, short_line, long_line, index_):
        long_is_lower = long_line.iloc[index_-1]<short_line.iloc[index_-1]
        long_is_upper = long_line.iloc[index_]>=short_line.iloc[index_]
        sellable = long_is_upper and long_is_lower
        return sellable

        
        
    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        _,_,_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month)
        prf_list = []
        is_bought = False
        index_buy = 0
        prf = 0
        trade_count = 0
        eval_price = 0
        total_eval_price = 0
        short_line = df_con['ma_short']
        long_line = df_con['ma_long']
        length = len(df_con)

        for i in range(1,length):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            if not is_bought:
                
                if self.is_buyable(short_line,long_line,i):
                    index_buy = df_con['close'].iloc[i]
                    is_bought = True
                    start_time = df_con.index[i]
                    hold_count_day = 0
                else:
                    continue
            
            
            else:
                
                if self.is_sellable(short_line,long_line,i) or hold_count_day==self.hold_day:
                    index_cell = df_con['close'].iloc[i]
                    end_time = df_con.index[i]
                    prf += index_cell - index_buy
                    prf_list.append(index_cell - index_buy)
                    total_eval_price = prf
                    self.pr_log['reward'].loc[df_con.index[i]] = prf 
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                    trade_count+=1
                    is_bought = False
                    hold_count_day = 0
                    pl.add_span(start_time,end_time)
                else:
                    hold_count_day+=1
                    eval_price = df_con['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                    
        
        if is_bought:
            end_time = df_con['close'].index[-1]
            index_sell = df_con['close'].iloc[-1]
            pl.add_span(start_time,end_time)
            eval_price = index_sell - index_buy
            prf += eval_price
            prf_list.append(prf)
            total_eval_price += eval_price
            self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        
        prf_array = np.array(prf_list)
        log = self.return_trade_log(prf,trade_count,prf_array,0)
        self.trade_log = log

        if not is_validate:        
            print(log)
            print("")
            pl.show()    

class FFTSimulation(XGBSimulation2):


    def __init__(self, lx, Fstrategies, alpha=0.33,width=20,window_type='none',is_high_pass=False,is_low_pass=True,is_ceps=False,cut_off=3):
        super().__init__(lx,alpha)
        self.Fstrategies = Fstrategies
        self.width = width
        self.window_type = window_type
        self.is_high_pass = is_high_pass
        self.is_low_pass = is_low_pass
        self.is_ceps = is_ceps
        self.cut_off = cut_off
        

    def choose_strategy(self,x_spe):
        cos_sim_list = []
        for fs in self.Fstrategies:
            cos = cos_sim(fs.spectrum,x_spe)
            cos_sim_list.append(cos)

        max_index = np.argmax(np.array(cos_sim_list))
        strategy = self.Fstrategies[max_index].strategy
        self.alpha = self.Fstrategies[max_index].alpha
        return strategy


    def make_x_dict(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        x_dict = {}
        lc = LearnClustering()
        x_, z_ = lc.make_x_data(df_con['close'],stride=1,test_rate=1.0,width=self.width)
        length = len(z_)

        for i in range(length):
            time_ = z_[i].index[-1]
            x_dict[time_] = standarize(x_[i])
        
        return x_dict


    def hanning(self,data, Fs):
        han = signal.hann(Fs)                    # ハニング窓作成
        acf = 1 / (sum(han) / Fs)                # 振幅補正係数(Amplitude Correction Factor)
        # オーバーラップされた複数時間波形全てに窓関数をかける
        data = data * han  # 窓関数をかける 
        return data, acf
    
    def hamming(self,data,Fs):
        ham = signal.hamming(Fs)
        acf = 1 / (sum(ham) / Fs)               
        data = data * ham  # 窓関数をかける 
        return data, acf
    
    def blackman(self,data,Fs):
        bla = signal.blackman(Fs)
        acf = 1 / (sum(bla) / Fs)               
        data = data * bla  # 窓関数をかける 
        return data, acf
    
    def butter_lowpass(self,lowcut, fs, order=4):
        '''
        バターワースローパスフィルタを設計する関数
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = signal.butter(order, low, btype='low')
        return b, a
    
    def butter_lowpass_filter(self, x, lowcut, fs, order=4):
        '''データにローパスフィルタをかける関数
        '''
        b, a = self.butter_lowpass(lowcut, fs, order=order)
        y = signal.filtfilt(b, a, x)
        return y
    
    def butter_highpass(self, highcut, fs, order=4):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = signal.butter(order, high, btype = "high", analog = False)
        return b, a

    def butter_highpass_filter(self, x, highcut, fs, order=4):
        b, a = self.butter_highpass(highcut, fs, order=order)
        y = signal.filtfilt(b, a, x)
        return y

    def do_fft(self,wave_vec):
        N = len(wave_vec)            # サンプル数
        dt = 0.1          # サンプリング間隔
        fs = 1/dt
        t = np.arange(0, N*dt, dt) # 時間軸
        freq = np.linspace(0, 1.0/dt, N) # 周波数軸
        cut_off = self.cut_off
        
        if self.is_high_pass:
            wave_vec = self.butter_highpass_filter(wave_vec,cut_off,fs)
        if self.is_low_pass:
            wave_vec = self.butter_lowpass_filter(wave_vec,cut_off,fs)
        

        if self.window_type=='han':
            f, acf = self.hanning(wave_vec,N)
        elif self.window_type=='ham':
            f, acf = self.hamming(wave_vec,N)
        elif self.window_type=='bla':
            f, acf = self.blackman(wave_vec,N)
        elif self.window_type=='none':
            f = wave_vec
            acf = 1
        else:
            print('Error')
            return
        
        F = np.fft.fft(f)
        F = F[:len(F)//2]
        # 振幅スペクトルを計算
        Amp = acf*np.abs(F/(N/2))
        return F, Amp

    def make_spectrum(self,wave_vec):
        F, Amp = self.do_fft(wave_vec)
        spectrum = Amp**2
        return standarize(spectrum)
    
    def db(self, x, dBref):
        delta = 10**-7
        x += delta
        y = 20 * np.log10(x / dBref)                      # リニア値をdB値に変換
        return y    
    
    def make_cepstrum(self,spectrum):
        spectrum += np.abs(np.min(spectrum))
        spec_db = self.db(spectrum, 2e-5)                              # スペクトルを対数(dB)にする(0dB=20[μPa])
        ceps_db = np.real(fftpack.ifft(spec_db))                # 対数スペクトルを逆フーリエ変換してケプストラム波形を作る
        ceps_db_low = fftpack.fft(ceps_db) 
        ceps_norm = norm(ceps_db_low)                           # ケプストラム波形を再度フーリエ変換してスペクトル包絡を得る
        length = len(ceps_norm)
        ceps_norm = np.abs(ceps_norm[:length//2])
        return standarize(ceps_norm)
        

    def simulate(self, path_tpx, path_daw, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        x_dict = self.make_x_dict(path_tpx,path_daw)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        self.predict_proba = predict_proba
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0
        # for debug
        self.strategies = []
        self.spe_list = []
        self.cnt_normal = 0
        self.cnt_reverse = 0

        
        for i in range(length-1):

            time_ = df_con.index[i]
            x_spe = self.make_spectrum(x_dict[time_])
            if self.is_ceps:
                x_spe = self.make_cepstrum(x_spe)
            self.spe_list.append(x_spe)

            if not is_bought:
                strategy = self.choose_strategy(x_spe)

            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)


            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2

            if strategy=='reverse':
                self.strategies.append(-1)
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
                if is_buy: self.cnt_reverse += 1
            elif strategy=='normal':
                self.strategies.append(1)
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
                if is_buy: self.cnt_normal += 1
            elif strategy=='stay' :
                self.strategies.append(0)
                is_buy = False
                is_sell =  False
                is_cant_buy = False

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            # print("buy_count",buy_count)
            # print("sell_count",sell_count)
            pl.show()

class FFTSimulation2(FFTSimulation):


    def __init__(self,Fstrategies):
        self.MK = MakeTrainData
        self.Fstrategies = Fstrategies
        self.ma_long =25
        self.ma_short=5

    
    def simulate(self, path_tpx, path_daw, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        x_dict = self.make_x_dict(path_tpx,path_daw)
        length = len(x_check)
        prf_list = []
        # predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0

        
        for i in range(length-1):

            time_ = df_con.index[i]
            x_spe = self.make_spectrum(x_dict[time_])

            # if not is_bought:
            strategy = self.choose_strategy(x_spe)

            # row = predict_proba[i]
            # label = np.argmax(row)
            # prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   


            if strategy=='normal':
                is_buy  = True
                is_sell = False
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            elif strategy=='stay' :
                is_buy = False
                is_sell =  True
                is_cant_buy = False

            

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                # is_triggerはあった方がいい
                
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        # self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        # df = self.calc_acc(acc_df, y_check)
        # self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            # print(df)
            print("")
            print("trigger_count :",trigger_count)
            # print("buy_count",buy_count)
            # print("sell_count",sell_count)
            pl.show()

class UpDownSimulation(Simulation):
    
    
    def __init__(self,lx,Ustrategies,width=40,alpha=0.34):
        super().__init__()
        self.lx = lx 
        self.xgb_model = lx.model
        self.Ustrategies = Ustrategies
        self.width = width
        self.alpha = alpha
        self.ma_long = 25
        self.ma_short = 5
    
    
    def choose_strategy(self,wave,cut_off=3,order=3):
        filtered_wave = butter_lowpass_filter(wave,cut_off,20,order=order)
        diff_wave = np.diff(filtered_wave)
        sum_diff = sum(diff_wave)
        strategy = 'none'

        # 右肩下がりの時
        if sum_diff<=0:
            strategy = 'stay'
    
        # 右肩上がりの時
        else:
            normal_wave = self.Ustrategies[0].spectrum
            reverse_wave = self.Ustrategies[2].spectrum
            
            n_cos = cos_sim(normal_wave,wave)
            r_cos = cos_sim(reverse_wave,wave)
            if n_cos>=r_cos:
                strategy = 'normal'
            else:
                strategy = 'reverse'
                
        return strategy
    
    def make_x_dict(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        x_dict = {}
        lc = LearnClustering()
        x_, z_ = lc.make_x_data(df_con['close'],stride=1,test_rate=1.0,width=self.width)
        length = len(z_)

        for i in range(length):
            time_ = z_[i].index[-1]
            x_dict[time_] = standarize(x_[i])
        
        return x_dict
    

    def simulate(self, path_tpx, path_daw, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
        is_observed=False,cut_off=3,order=4):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        x_dict = self.make_x_dict(path_tpx,path_daw)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        self.predict_proba = predict_proba
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0
        # for debug
        self.strategies = []
        self.spe_list = []
        self.cnt_normal = 0
        self.cnt_reverse = 0
        
        for i in range(length-1):

            time_ = df_con.index[i]
            wave = x_dict[time_]

            if not is_bought:
                strategy = self.choose_strategy(wave,cut_off=cut_off,order=order)

            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)


            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2

            if strategy=='reverse':
                self.strategies.append(-1)
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
                if is_buy: self.cnt_reverse += 1
            elif strategy=='normal':
                self.strategies.append(1)
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)
                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
                if is_buy: self.cnt_normal += 1
            elif strategy=='stay' :
                self.strategies.append(0)
                is_buy = False
                is_sell =  False
                is_cant_buy = False

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            # print("buy_count",buy_count)
            # print("sell_count",sell_count)
            pl.show()
            
class ClusterSimulation(FFTSimulation):


    def __init__(self,lx,Cstrategies,width=20):
        super().__init__(lx,Cstrategies,width=width)

    
    def simulate(self, path_tpx, path_daw, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False):
        
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        x_tmp,y_tmp,current_date,acc_df = self.set_for_online(x_check,y_)
        x_dict = self.make_x_dict(path_tpx,path_daw)
        length = len(x_check)
        prf_list = []
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0

        
        for i in range(length-1):

            time_ = df_con.index[i]
            x_spe = x_dict[time_]

            if not is_bought:
                strategy = self.choose_strategy(x_spe)

            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            tmp_date = x_tmp.index[i]   

            if is_online and current_date.month!=tmp_date.month:
                predict_proba, current_date = self.learn_online(x_tmp,y_tmp,x_check,current_date,tmp_date)


            if prob > self.alpha:

                if label == 0:
                    acc_df.iloc[i] = 0
                elif label == 1:  
                    acc_df.iloc[i] = 1
                else: #  label==2
                    acc_df.iloc[i] = 2

            if strategy=='reverse':
                is_buy  = (label==0 and prob>self.alpha)
                is_sell = (label==2 and prob>self.alpha)

                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]))
            elif strategy=='normal':
                is_buy  = (label==2 and prob>self.alpha)
                is_sell = (label==0 and prob>self.alpha)

                is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))
            elif strategy=='stay' :
                is_buy = False
                is_sell =  False
                is_cant_buy = False

            

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        # try:
        df = self.calc_acc(acc_df, y_check)
        self.accuracy_df = df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print(df)
            print("")
            print("trigger_count :",trigger_count)
            # print("buy_count",buy_count)
            # print("sell_count",sell_count)
            pl.show()

class CeilSimulation(Simulation):


    def __init__(self,alpha=0.8,beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.MK = MakeTrainData


    def make_z_dict(self,path_tpx,path_daw,width=20,stride=1,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        z_dict = {}
        lc = LearnClustering()
        _, z_ = lc.make_x_data(df_con['close'],width=width,stride=stride,test_rate=test_rate)
        length = len(z_)

        for i in range(length):
            time_ = z_[i].index[-1]
            z_dict[time_] = z_[i]
        
        return z_dict


    def calc_ceil(self,z_,close_):
        L = z_.min()
        H = z_.max()
        ceil_ = (close_ - L)/(H - L)
        return ceil_


    def simulate(self,path_tpx, path_daw, is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False,width=20,stride=1):
        x_check,y_check,y_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,'None',is_validate)
        self.x_check = x_check
        z_dict = self.make_z_dict(path_tpx,path_daw,width=width,stride=stride)
        length = len(x_check)
        prf_list = []
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0
        ceil_list = []
        self.x_check = x_check

        for i in range(length-1):

            time_ = df_con.index[i]
            z_ = z_dict[time_]
            close_ = df_con['close'].iloc[i]
            ceil_ = self.calc_ceil(z_,close_)
            ceil_list.append(ceil_)

            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price

            # 底で買って, 天井で売る
            is_buy  = ceil_<self.beta
            is_sell = ceil_>self.alpha
            is_cant_buy = (is_observed and (df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]))

            
            if not is_bought:
                if is_cant_buy or self.wallet+prf < index_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df_con,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df_con,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        ceil_df = pd.DataFrame(ceil_list,columns={'ceil'},index=x_check.index[:-1])
        self.ceil_df = ceil_df
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.y_check = y_check
        self.ceil_df = ceil_df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print("trigger_count :",trigger_count)
            pl.show()


    def show_ceil_chart(self):
        plt.clf()
        chart_ = self.df_con['close']
        ceil_df = self.ceil_df.copy()
        scale = chart_.mean() * 0.9
        _, ax = plt.subplots(figsize=(20, 6))
        ax.plot(chart_.iloc[:-1],label='close')
        ax.plot(ceil_df['ceil']*scale,label='ceil')
        plt.grid()
        plt.show()

class RandomSimulation(Simulation):


    def __init__(self,ma_short=5,ma_long=25,random_num=2):
        super().__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.random_num = random_num


    def make_check_data(self,path_tpx,path_daw):
        df = self.make_df_con(path_tpx,path_daw)
        x_check = df.iloc[1:]
        length = len(df)
        y_check = []
        for i in range(1,length):
            if df['pclose'].iloc[i] > 0:
                y_check.append(1)
            else:
                y_check.append(0)
        
        return x_check, y_check

    def random_func(self,random_num):
        # 0 or 1 を返す関数
        return random.randint(0,random_num-1)


    # ランダムに上がるか, 下がるか予測する
    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        prf_list = []
        trade_count = 0
        df_con = self.return_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        pl.add_plot(df_con['open'],label='open')
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
        #********* acc_df?
        acc_df = pd.DataFrame(index=x_check.index)
        acc_df['pred'] = [-1] * len(acc_df)
#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
# is_observed=True としたことで買えなくなった取引の回数をカウント
        cant_buy = 0
        pclose = x_check['pclose']


        for i in range(length-1):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            #*******  self.random_num で　up, down(あるいはstay） を 返す関数を実装
            label = self.random_func(self.random_num)

            if label==1 and pclose.iloc[i+1]>0:
                acc_df.iloc[i] = 1
            else: #label == 0 
                acc_df.iloc[i] = 0
                # x_check['dclose'].iloc[i] は観測可能 

            if not is_bought:
                # 買いのサイン
                if label==1:
                    index_buy = df_con['close'].loc[x_check.index[i+1]]
                    start_time = x_check.index[i+1]
                    is_bought = True
            else:
                # 売りのサイン
                if label==0:
                    index_sell = df_con['close'].loc[x_check.index[i+1]]
                    end_time = x_check.index[i+1]
                    prf += index_sell - index_buy
                    prf_list.append(index_sell - index_buy)
                    is_bought = False
                    trade_count += 1
                    pl.add_span(start_time,end_time)
                else:
                    eval_price = df_con['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price

            
            self.is_bought = is_bought
                  
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                # ここも df 化できるように 
                print(log)
                print("")
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")

# alpha, beta よくよく働きを吟味するように
# まだ仕組み理解してない
class DawSimulation(Simulation):

    def __init__(self,alpha=0,beta=0):
        super().__init__()
        # 買いの閾値をalpha
        self.alpha = alpha
        # 売りの閾値をbeta
        self.beta = beta
        self.ma_short = 5
        self.ma_long = 25


    def make_check_data(self,path_tpx,path_daw):
        df = self.make_df_con(path_tpx,path_daw)
        x_check = df.iloc[1:]
        length = len(df)
        y_check = []
        for i in range(1,length):
            if df['pclose'].iloc[i] > 0:
                y_check.append(1)
            else:
                y_check.append(0)
        
        return x_check, y_check



    def simulate(self, path_tpx, path_daw, is_validate=False,strategy='normal',start_year=2021,end_year=2021,start_month=1,end_month=12,
                 is_variable_strategy=False,is_observed=False):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        trade_count = 0
        df_con = self.make_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        pl.add_plot(df_con['open'],label='open')
        prf_list = []
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0

        #********* acc_df?
        acc_df = pd.DataFrame(index=x_check.index)
        acc_df['pred'] = [-1] * len(acc_df)


#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
# is_observed=True としたことで買えなくなった取引の回数をカウント
        cant_buy = 0
        dclose = x_check['dclose']
        pclose = x_check['pclose']


        for i in range(length-1):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            
            if is_variable_strategy:
                strategy = self.return_grad(df_con, index=i,gamma=0, delta=0)
            

#   ダウが上がる　-> 日経平均もあがることを仮定している
# 　つまり, dclose>0 -> label = 1
            if dclose.iloc[i+1]>0:
                acc_df.iloc[i] = 1
            else: #label == 0
                acc_df.iloc[i] = 0
                # x_check['dclose'].iloc[i] は観測可能 


            if strategy=='reverse':
            
                if not is_bought:
    #                 下がって買い
    # ダウ平均の変化率(%)が下がって買う -> 逆張り戦略
                    if x_check['dclose'].iloc[i]*100 < self.alpha:
#                         観測した始値が, 下がるという予測に反して上がっていた時, 買わない
                        if is_observed and df_con['open'].loc[x_check.index[i+1]] > df_con['close'].loc[x_check.index[i]]:
                            cant_buy += 1
                            continue
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                        
#                     
                else:
    #                 上がって売り
                    if x_check['dclose'].iloc[i]*100 > self.beta:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                        
                        
            elif strategy=='normal':
                
                if not is_bought:
    #                 上がって買い
                    if x_check['dclose'].iloc[i]*100 > self.alpha:
#             上がるという予測に反して, 始値が前日終値より下がっていたら買わない
                        if is_observed and df_con['open'].loc[x_check.index[i+1]] < df_con['close'].loc[x_check.index[i]]:
                            cant_buy += 1
                            continue
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 下がって売り
                    if x_check['dclose'].iloc[i]*100 < self.beta:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            else:
                print("No such strategy.")
                return 
            
            self.is_bought = is_bought

        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            df = self.calc_acc(acc_df, y_check)
            self.accuracy_df = df
            log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
            self.trade_log = log

            if not is_validate:
                print(log)
                print("")
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")

class TPXSimulation(Simulation):


    def __init__(self):
        super().__init__()


    def simulate(self,path_tpx,path_daw,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12,df_='None'):
        _,_,_,df_con,pl = self.simulate_routine(path_tpx, path_daw,start_year,end_year,start_month,end_month,df_,is_validate)
        length = len(df_con)
        prf_list = []
        index_buy = 0
        prf = 0
        start_time = df_con.index[0]
        end_time = df_con.index[-1]
        if not is_validate:
            pl.add_span(start_time,end_time)
        prf = df_con['close'].loc[df_con.index[-1]] - df_con['close'].loc[df_con.index[0]]
        index_buy = df_con['close'].iloc[0]
        prf_list_diff = df_con['close'].map(lambda x : x - index_buy).diff().fillna(0).tolist()
        prf_list = df_con['close'].map(lambda x : x - index_buy).tolist()
        prf_array_diff = np.array(prf_list_diff)
        self.pr_log = pd.DataFrame(index=df_con.index[:-1])
        self.pr_log['reward'] = prf_list[:-1]
        self.pr_log['eval_reward'] = prf_list[:-1]
        log = self.return_trade_log(prf,length-1,prf_array_diff,0)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            pl.show()

class StrategymakerSimulation(XGBSimulation):


    def simulate(self, path_tpx, path_daw, sm, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,ma_short=5,ma_long=25,theta=0.0001):
        x_check, y_check = self.make_check_data(path_tpx,path_daw)
        y_ = pd.DataFrame(y_check)
        y_.index = x_check.index
        x_check = self.return_split_df(x_check,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_ = self.return_split_df(y_,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        y_check = y_.values.reshape(-1).tolist()
        length = len(x_check)
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        trade_count = 0
        df_con = self.make_df_con(path_tpx,path_daw)
        df_con['ma_short'] = df_con['close'].rolling(self.ma_short).mean()
        df_con['ma_long']  = df_con['close'].rolling(self.ma_long).mean()
        df_con = df_con.iloc[self.ma_long:-1]
        df_con = self.return_split_df(df_con,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        pl = PlotTrade(df_con['close'],label='close')
        pl.add_plot(df_con['ma_short'],label='ma_short')
        pl.add_plot(df_con['ma_long'],label='ma_long')
        prf_list = []
        self.pr_log = pd.DataFrame(index=x_check.index[:-1])
        self.pr_log.index = x_check.index
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()
        eval_price = 0
        total_eval_price = 0
#*      オンライン学習用の学習データ   
        x_tmp = x_check.copy()
        y_tmp = y_.copy()
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame()
        acc_df.index = x_tmp.index
        acc_df['pred'] = [-1] * len(acc_df)
        x_sm, y_sm = sm.make_train_data(path_tpx, path_daw,year=start_year,theta=theta)
        x_sm = self.return_split_df(x_sm,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)
        buy_sign = sm.model.predict(x_sm)
#* 判定不能は -1, 騰貴予測は 1, 下落予測は 0
        
        
        for i in range(length-1):
            
            
            row = predict_proba[i]
            label = np.argmax(row)
            prob = row[label]
            total_eval_price = prf
            
            self.pr_log['reward'].loc[df_con.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
#             label==0 -> down
#             label==1 -> up
#*          オンライン学習
            tmp_date = x_tmp.index[i]
            if current_date.month!=tmp_date.month and is_online:
#             x_ = x_tmp.loc[:x_tmp.index]
                x_ = x_tmp[current_date<=x_tmp.index]
                x_ = x_[x_.index<tmp_date]
                y_ = y_tmp[current_date<=y_tmp.index]
                y_ = y_[y_.index<tmp_date]
#                 param_dist = {'objective':'binary:logistic', 'n_estimators':16,'use_label_encoder':False,
#                  'max_depth':4}
#                 tmp_xgb = xgb.XGBClassifier(**param_dist)
                self.xgb_model = self.xgb_model.fit(x_,y_)
                predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
                current_date = tmp_date
            
            if prob > self.alpha:
                if label == 0:
                    acc_df.iloc[i] = 0
                else: #l able == 1 
                    acc_df.iloc[i] = 1
                    
#                     「買い」 サインの時
            if buy_sign[i]==1:
                if prob >0.5 :
                    strategy='normal'
                else:
                    strategy='reverse'
            else:
                strategy=None
                

            if strategy=='reverse' and buy_sign[i]==1:
            
                if not is_bought:
    #                 下がって買い
                    if label==0 and prob>self.alpha:
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 上がって売り
                    if label==1 and prob>self.alpha:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
                        
                        
            elif strategy=='normal' and buy_sign[i]==1:
                
                if not is_bought:
    #                 上がって買い
                    if label==1 and prob>self.alpha:
                        index_buy = df_con['close'].loc[x_check.index[i+1]]
                        start_time = x_check.index[i+1]
                        is_bought = True
                else:
    #                 下がって売り
                    if label==0 and prob>self.alpha:
                        index_sell = df_con['close'].loc[x_check.index[i+1]]
                        end_time = x_check.index[i+1]
                        prf += index_sell - index_buy
                        prf_list.append(index_sell - index_buy)
                        is_bought = False
                        trade_count += 1
                        pl.add_span(start_time,end_time)
                    else:
                        eval_price = df_con['close'].iloc[i] - index_buy
                        total_eval_price += eval_price
                        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
            
            
            elif strategy==None:
                continue
                
        
        
        if is_bought:
            index_sell = df_con['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            pl.add_span(start_time,end_time)

        
        self.pr_log['reward'].loc[df_con.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df_con.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.acc_df = acc_df
        self.y_check = y_check
            
        
        try:
            if not is_validate:
                print("Total profit :{}".format(prf))
                print("Trade count  :{}".format(trade_count))
                print("Max profit   :{}".format(prf_array.max()))
                print("Min profit   :{}".format(prf_array.min()))
                print("Mean profit  :{}".format(prf_array.mean()))
                if not is_online:
                    df = self.eval_proba(x_check,y_check)
                else:
                    df = self.calc_acc(acc_df, y_check)
                print(df)
                print("")
                pl.show()
        except:
            print("no trade")

