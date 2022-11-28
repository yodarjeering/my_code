import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_corr(x_):
    x_corr = x_.corr()
    fig, ax = plt.subplots(figsize=(20, 20)) 
    sns.heatmap(x_corr, square=True, vmax=1, vmin=-1, center=0)
    plt.plot()
    
    
def return_corr(df,year=2021):
    x = df[df.index.year==year]['daw_close'].values
    y = df[df.index.year==year]['close'].values
    corr = np.corrcoef(x,y)
    return corr


def return_strong_corr(x_):
    strong_corr = []
    x_corr = x_.corr()
    for idx in x_corr.index:
        for col in x_corr.columns:
            if idx == col:
                continue
            else:
                corr = x_corr.loc[idx][col]
                if abs(corr)>=0.8:
                    strong_corr.append([idx,col])
    return strong_corr