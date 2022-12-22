import numpy as np
import xgboost as xgb 
from sklearn.metrics import classification_report,roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from modules.preprocessing import MakeTrainData,MakeTrainData3
from modules.preprocessing import DataFramePreProcessing

from modules.funcs import standarize
import statsmodels.api as sm
from sklearn import preprocessing


def make_data(x_,y_):
    train = pd.concat([y_,x_],axis = 1,join='inner').astype(float)
    x_ = train[train.columns[1:]]
    y_ = train[train.columns[0]]
    return x_,y_


class LearnXGB():
    
    
    def __init__(self,num_class=2):
        self.model = xgb.XGBClassifier()
        self.x_test = None
        self.num_class = num_class
        if num_class==2:
            self.MK = MakeTrainData
        else: # num_class==3
            self.MK = MakeTrainData3
    

    def learn_xgb(self, path_tpx, path_daw, test_rate=0.8, param_dist='None',verbose=True):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
            'n_estimators':16,
            'max_depth':4,
            'random_state':0
            }

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if verbose:
            print("---------------------")
            if self.num_class==2:
                print('AUC train:',roc_auc_score(y_train,y_proba_train))    
                print('AUC test :',roc_auc_score(y_test,y_proba))

            print(classification_report(np.array(y_test), hr_pred))
            _, ax = plt.subplots(figsize=(12, 10))
            xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model


    def learn_xgb2(self,x_train,y_train,x_test,y_test,param_dist='None',verbose=True):
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
                'n_estimators':16,
                'max_depth':4,
                'random_state':0
                }

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if verbose:
            print("---------------------")
            if self.num_class==2:
                print('AUC train:',roc_auc_score(y_train,y_proba_train))    
                print('AUC test :',roc_auc_score(y_test,y_proba))

            print(classification_report(np.array(y_test), hr_pred))
            _, ax = plt.subplots(figsize=(12, 10))
            xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model
        

    def make_state(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))
        chart_ = df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        return state_, chart_
        
        
    def make_xgb_data(self, path_tpx, path_daw, test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=test_rate)
        x_train, y_train, x_test, y_test = mk.make_data()
        return x_train,y_train,x_test,y_test
    
    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose']],axis = 1,join='inner').astype(float)
        df_con['pclose'] = df_con['close'].pct_change()
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con
    
    
    def for_ql_data(self, path_tpx, path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)

        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))

        chart_ = mk.df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        state_ = pd.DataFrame(state_)
        state_['day'] = chart_.index
        
        state_.reset_index(inplace=True)
        state_.set_index('day',inplace=True)
        state_.drop('index',axis=1,inplace=True)
        return state_, chart_
    
    ## ここ変える
    ## XGBSimulation 入力する感じか, 新しい関数
    def predict_tomorrow(self, path_tpx, path_daw, alpha=0.5, strategy='normal', is_online=False, is_valiable_strategy=False,start_year=2021,start_month=1,end_month=12,is_observed=False,is_validate=False):
        xl = XGBSimulation(self,alpha=alpha)
        xl.simulate(path_tpx,path_daw,is_validate=is_validate,strategy=strategy,is_variable_strategy=is_valiable_strategy,start_year=start_year,start_month=start_month,end_month=end_month,is_observed=is_observed,is_online=is_online)
        self.xl = xl
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        tomorrow_predict = self.model.predict_proba(x_check)
        label = self.get_tomorrow_label(tomorrow_predict,strategy, is_valiable_strategy)
        print("is_bought",xl.is_bought)
        print("df_con in predict_tomorrow",df_con.index[-1])
        print("today :",x_check.index[-1])
        print("tomorrow UP possibility", tomorrow_predict[-1,1])
        print("label :",label)


    def get_tomorrow_label(self, tomorrow_predict,strategy, is_valiable_strategy):
        label = "STAY"
        df_con = self.xl.df_con
        if is_valiable_strategy:
            i = len(df_con)-2
            strategy = self.xl.return_grad(df_con, index=i,gamma=0, delta=0)
        
        if strategy == 'normal':
            if tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label =  "SELL"
            else:
                label = "STAY"
        
        elif strategy == 'reverse':
            if 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif tomorrow_predict[-1,1] > self.xl.alpha:
                label = "SELL"
            else:
                label = "STAY"

        return label


class LearnClustering(LearnXGB):


    def __init__(self,n_cluster=8,random_state=0,width=40,stride=5,strategy_table=None):
        super().__init__()
        self.model : KMeans = None
        self.n_cluster = n_cluster
        self.width = width
        self.stride = stride
        self.n_label = None
        self.wave_dict = None
        self.strategy_table = strategy_table
        self.random_state=random_state



    def make_x_data(self,close_,width=20,stride=5,test_rate=0.8):
        length = int(len(close_)*test_rate)
        close_ = close_.iloc[:length]


        x = []
        z = []
        for i in range(0,length-width,stride):
            x.append(standarize(close_.iloc[i:i+width]).tolist())
            z.append(close_.iloc[i:i+width])
        x = np.array(x)
        return x,z


    def make_wave_dict(self,x,y,width):
        n_label = list(set(y))
        self.n_label = n_label
        wave_dict = {i:np.array([0.0 for j in range(width)]) for i in n_label}
        
        # クラス波形の総和
        for i in range(len(x)):
            wave_dict[y[i]] += x[i]
        
        # 平均クラス波形
        for i in range(len(y)):
            count_class = list(y).count(y[i])
            wave_dict[y[i]] /= count_class
            wave_dict[y[i]] = preprocessing.scale(wave_dict[y[i]])
        return wave_dict


    def learn_clustering(self,path_tpx,path_daw,width=20,stride=5,test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con['close']
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=test_rate)
        model = KMeans(n_clusters=self.n_cluster,random_state=self.random_state)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering2(self,close_,width=20,stride=5):
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=1.0)
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering3(self,x,width=20):
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict
    

    def show_class_wave(self):
        for i in range(self.n_cluster):
            print("--------------------")
            print("class :",i)
            plt.plot(self.wave_dict[i])
            plt.show()
            plt.clf()


    def predict(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z


    def predict2(self,df_con,stride=2,test_rate=1.0):
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z

    
    def return_y_pred(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred

    def encode(self, strategy, alpha, wave_dict):
        pass


class LearnTree(LearnXGB):
    
    
    def __init__(self,num_class=2):
        super().__init__(num_class=num_class)
        self.model : tree.DecisionTreeClassifier() = None
        self.x_test = None
    
    
    def learn_tree(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        tree_model = tree.DecisionTreeClassifier(random_state=0)
        hr_pred = tree_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = tree_model.predict_proba(x_train)[:,1]
        y_proba = tree_model.predict_proba(x_test)[:,1]
        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = tree_model
        
class LearnRandomForest(LearnXGB):
    
    
    def __init__(self,num_class=2):
        super().__init__(num_class=num_class)
        self.model : RandomForestClassifier = None
        self.x_test = None
    
    
    def learn_forest(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        tree_model = self.model = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=15)
        hr_pred = tree_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = tree_model.predict_proba(x_train)[:,1]
        y_proba = tree_model.predict_proba(x_test)[:,1]
        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = tree_model

class LearnLogisticRegressor(LearnXGB):
    
    
    def __init__(self,num_class):
        super().__init__(num_class=num_class)
        self.model : LogisticRegression = None
        self.x_test = None
    
    
    def learn_logistic(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        logistic_model = LogisticRegression(max_iter=2000)
        hr_pred = logistic_model.fit(x_train.astype(float), np.array(y_train)).predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = logistic_model.predict_proba(x_train)[:,1]
        y_proba = logistic_model.predict_proba(x_test)[:,1]
        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))
        print(classification_report(np.array(y_test), hr_pred))
        self.model = logistic_model

class LearnLinearRegression(LearnXGB):


    def __init__(self):
        super().__init__()
        self.model : LinearRegression = None
        self.x_test = None
        self.x_val = None
        self.y_val = None
        plt.clf()

    # データリーク確認
    # 直すように
    # diff_date : 何日後の予測をするか指定するパラメタ
    def make_regression_data(self,path_tpx,path_daw,test_rate=0.8,diff_date=5):
        df = self.make_df_con(path_tpx,path_daw)
        x_train,_,x_test,_ = self.make_xgb_data(path_tpx,path_daw,test_rate=test_rate)
        x_train, y_train = make_data(x_train,df['close'])
        x_test,y_test = make_data(x_test,df['close'])
        x_train = standarize(x_train)
        x_test = standarize(x_test)
        y_train = standarize(y_train)
        y_test = standarize(y_test)
        x_train = x_train.iloc[:-diff_date]
        y_train = y_train.iloc[diff_date:]
        x_test = x_test.iloc[:-diff_date]
        y_test = y_test.iloc[diff_date:]
        self.x_val = x_test
        self.y_val = y_test
        return x_train,y_train,x_test,y_test



    def learn_linear_regression(self,path_tpx,path_daw,test_rate=0.8):
        x_train,y_train,x_test,y_test = self.make_regression_data(path_tpx,path_daw,test_rate=test_rate)
        lr = LinearRegression()
        lr.fit(x_train,y_train)
        self.model = lr
        y_pred = lr.predict(x_test)
        print("True")
        plt.plot(y_test.iloc[:20])
        plt.show()
        print("Predict")
        plt.plot(y_pred[:20])
        plt.grid()
        plt.show()


    def show_model_summary(self):
        x_add_const = sm.add_constant(self.x_val)
        model_sm = sm.OLS(self.y_val, x_add_const).fit()
        print(model_sm.summary())

