import numpy as np
# from modules.training import LearnClustering
# from modules.simulation import XGBSimulation2
from modules.funcs import standarize
from modules.training import LearnClustering
from collections import namedtuple
from modules.simulation import XGBSimulation2
from modules.funcs import butter_lowpass_filter, butter_lowpass

ValueTable = namedtuple("ValueTable",["strategy","alpha","total_profit","trade_log","stock_wave"])
Fstrategy = namedtuple("Fstrategy",["strategy","alpha","spectrum"])



def make_easy_x(ng):
    x = []
    for df in ng:
        lis_ =  df.tolist()
        # print(lis_)
        x.append(lis_)
    x = np.array(x)
    return x

def make_value_list(lx,start_year,end_year,path_tpx,path_daw,alpha=0.34,width=20,stride=10,start_month=1,end_month=12):

    lc_dummy = LearnClustering(width=width)
    df_con = lc_dummy.make_df_con(path_tpx,path_daw)
    
    df_con = df_con[df_con.index.year<=end_year]
    df_con = df_con[df_con.index.year>=start_year]
    df_con = df_con[df_con.index.month<=end_month]
    df_con = df_con[df_con.index.month>=start_month]
    
    x_,z_ = lc_dummy.make_x_data(df_con['close'],stride=stride,test_rate=1.0,width=width)
    length = len(z_)
    value_list = []

    for i in range(length):
        for strategy in ['normal','reverse']:
            try:
                xl = XGBSimulation2(lx,alpha=alpha)
                xl.simulate(path_tpx,path_daw,strategy=strategy,is_validate=True,start_year=start_year,end_year=end_year,df_=z_[i])
                
                trade_log =  xl.trade_log
                total_profit = trade_log['total_profit'].values[0]
                stock_wave = z_[i]
                vt = ValueTable(strategy,alpha,total_profit,trade_log,stock_wave)
                value_list.append(vt)
                
            except Exception as e:
                print(e)
                continue

    return value_list

def return_clx(Value_list):
    Value_good = sorted(Value_list,key=lambda x :x[2],reverse=True)
    Value_bad = sorted(Value_list,key=lambda x :x[2],reverse=False)
    ng = []
    rg = []
    nb = []
    rb = []
    
    # 1sigam = 外れ値 として処理する
    prf_list=[]
    for vg in Value_good:
        total_profit = vg.total_profit
        prf_list.append(total_profit)      
    prf_array = np.array(prf_list)
    st_prf = standarize(prf_array)

    for idx,v in enumerate(Value_good):
        if v.total_profit<=0:break
        if np.abs(st_prf[idx]) >=1:continue    

        df = v.stock_wave
        strategy = v.strategy
        # print(df)
        # break
        if strategy=="normal":
            ng.append(standarize(df))
        else:
            rg.append(standarize(df))

    prf_list=[]
    for vb in Value_bad:
        total_profit = vb.total_profit
        prf_list.append(total_profit)      
    prf_array = np.array(prf_list)
    st_prf = standarize(prf_array)

    for v in Value_bad:
        if v.total_profit>=0 :break
        if np.abs(st_prf[idx]) >=1:continue  
        
        df = v.stock_wave
        strategy = v.strategy

        if strategy=="normal":
            nb.append(standarize(df))
        else:
            rb.append(standarize(df))

    x_ng = make_easy_x(ng)
    x_nb = make_easy_x(nb)
    x_rg = make_easy_x(rg)
    x_rb = make_easy_x(rb)
    return x_ng,x_nb,x_rg,x_rb

def return_ffs(lx,x_ng,x_nb,x_rg,x_rb,FFT_obj,width=20,stride=5):


    log_dict = {}
    cs_dict = {}
    ffs_dict = {}

    random_state=0

    alpha = 0.33
    n_cluster = 1
        
    Fstrategies = []
    Cstrategies = []
    Phases = []
    lc_rg = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rg.learn_clustering3(x_rg,width=width,stride=stride)
    lc_rb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rb.learn_clustering3(x_rb,width=width,stride=stride)
    lc_ng = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_ng.learn_clustering3(x_ng,width=width,stride=stride)
    lc_nb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_nb.learn_clustering3(x_nb,width=width,stride=stride)

    strategy_list = ['normal','stay','reverse','stay']

    j=0
    fft_dummy = FFT_obj(lx,None,width=width)
    for lc in [lc_ng,lc_nb,lc_rg,lc_rb]:
        
        for _,key in enumerate(lc.wave_dict):
            wave = lc.wave_dict[key]
            spe = fft_dummy.make_spectrum(wave)
            F,Amp = fft_dummy.do_fft(wave)
            F = F[:len(F)//2]
            phase = np.degrees(np.angle(F))
            strategy = strategy_list[j]
            fs  = Fstrategy(strategy,alpha,spe)
            cs = Fstrategy(strategy,alpha,wave)
            ph = Fstrategy(strategy,alpha,phase)
            Fstrategies.append(fs)
            Cstrategies.append(cs)
            Phases.append(ph)
        j+=1

    return Fstrategies,Phases

def return_fft_list(lx,x_,FFT_obj,width=20):

    fft_list = []
    fft_dummy = FFT_obj(lx,None,width=width)
    for wave in x_:
        spe = fft_dummy.make_spectrum(wave)
        fft_list.append(spe)
        
    return fft_list        

def return_uds_dict(value_dict,lx_dict,path_tpx,path_daw,width=40,stride=5,window_type='bla'\
    ,is_high_pass=False,is_low_pass=True,is_ceps=False,cut_off=3,order=3,limit_year=2016):
    uds_dict = {}
    trade_dict = {}
    lx_dummy = LearnXGB(num_class=3)
    F_list = []
    F_lis_dict = {}
    # limit_year = 2009
    alpha = 0.34

    for year in range(limit_year+2,2022):
        # print(year)
        start_month=1
        end_month = 12
        start_year = year
        end_year = year
        value_list = value_dict[str(year-1)]
        lx_ = lx_dict[str(year-1)]

        x_ng,x_nb,x_rg,x_rb = return_clx(value_list)

        try:
            Fstrategies= return_uds(lx_,x_ng,x_nb,x_rg,x_rb,UpDownSimulation,width=width,stride=stride,\
                cut_off=cut_off,order=order)
            """
            # Fstrategies をどんどん加算していく    
            # if len(uds_dict)>0:
            #     last_key = next(reversed(uds_dict),None)
            #     Fstrategies_old = uds_dict[last_key].Ustrategies
            #     Fstrategies = return_cumulative_fst(Fstrategies,Fstrategies_old)
            """
        except Exception as e:
            print(e)
            last_key = next(reversed(uds_dict),None)
            Fstrategies = uds_dict[last_key].Ustrategies
        
        # ffs = FFTSimulation(lx,Fstrategies,width=width,window_type=window_type)
        uds = UpDownSimulation(lx_,Fstrategies,width=width,alpha=alpha)
        uds.simulate(path_tpx,path_daw,start_year=year,end_year=year,is_validate=True,cut_off=cut_off,order=order)
        
        uds_dict[str(year)] = uds
        trade_dict[str(year)] = uds.trade_log
        F_lis_dict[str(year)] = F_list
    
    return uds_dict, trade_dict, F_lis_dict

def return_uds(lx,x_ng,x_nb,x_rg,x_rb,UpDownSimulation,width=20,stride=10,cut_off=3,order=4):

    log_dict = {}
    cs_dict = {}
    ffs_dict = {}
    random_state=0
    alpha = 0.33
    n_cluster = 1
    Fstrategies = []

    lc_rg = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rg.learn_clustering3(x_rg,width=width)
    lc_rb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_rb.learn_clustering3(x_rb,width=width)
    lc_ng = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_ng.learn_clustering3(x_ng,width=width)
    lc_nb = LearnClustering(n_cluster=n_cluster,random_state=random_state)
    lc_nb.learn_clustering3(x_nb,width=width)

    strategy_list = ['normal','stay','reverse','stay']

    j=0
    fft_dummy = UpDownSimulation(lx,None,width=width)
    for lc in [lc_ng,lc_nb,lc_rg,lc_rb]:
        
        for _,key in enumerate(lc.wave_dict):
            wave = lc.wave_dict[key]
            # このタイミングでハイパスフィルタかける
            # make_spectrum 内の関数をいじる
            filtered_wave = butter_lowpass_filter(wave,cut_off,20,order=order)
            strategy = strategy_list[j]
            fs  = Fstrategy(strategy,alpha,filtered_wave)
            Fstrategies.append(fs)

        j+=1

    return Fstrategies
