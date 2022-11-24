from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class AnalyzePCA():


    def __init__(self):
        self.pca : PCA = None
        self.data = None


    def make_data(self,path_tpx,path_daw,test_rate=0.8):
        lx = LearnXGB()
        x_train,y_train,x_test,y_test = lx.make_xgb_data(path_tpx,path_daw,test_rate=0.8)
        return x_train,y_train,x_test,y_test

    
    def do_pca(self,path_tpx,path_daw,test_rate=0.8):
        x_train,y_train,x_test,y_test = self.make_data(path_tpx,path_daw,test_rate=test_rate)
        x_train = standarize(x_train)
        x_test = standarize(x_test)
        pca = PCA()
        pca.fit(x_train)
        self.pca = pca
        self.data = x_train


    def do_pca2(self,x_train):
        pca = PCA()
        pca.fit(x_train)
        self.pca = pca
        self.data = x_train

    
    def get_loadings(self,is_show=False):
        loadings = pd.DataFrame(self.pca.components_.T, index=self.data.columns)
        if is_show:
            print(loadings.head())
        return loadings

    
    def get_score(self,is_show=False):
        score = pd.DataFrame(self.pca.transform(self.data), index=self.data.index)
        if is_show:
            print(score.head())
        return score

    # 第一主成分, 第二主成分に対するデータのプロット
    def show_data_in_k1k2(self,num=5):
        # num : 可視化するデータ数を指定
        plt.subplots(figsize=(10, 10)) 
        score = self.get_score()
        plt.scatter(score.iloc[:num,0], score.iloc[:num,1]) 
        plt.rcParams["font.size"] = 11
        # プロットしたデータにサンプル名をラベリング
        for i in range(num):
            plt.text(score.iloc[i,0], score.iloc[i,1], score.index[i], horizontalalignment="center", verticalalignment="bottom")
        # 第一主成分
        plt.xlim(-5, 2.5)
        # 第二主成分
        plt.ylim(-3, -1)
        plt.yticks(np.arange(-3, -0.5, 0.5))
        plt.xlabel("t1")
        plt.ylabel("t2")
        plt.grid()
        plt.show()
        plt.clf()
            

    def get_contribution_ratios(self,is_show=False):
        contribution_ratios = pd.DataFrame(self.pca.explained_variance_ratio_)
        if is_show:
            print(contribution_ratios.head())
        return contribution_ratios


    def get_cumulative_contribution_ratios(self,is_show=False):
        contribution_ratios = self.get_contribution_ratios()
        cumulative_contribution_ratios = contribution_ratios.cumsum()
        if is_show:
            print(cumulative_contribution_ratios)
        return cumulative_contribution_ratios


    def show_cont_cumcont_rartios(self):
        contribution_ratios = self.get_contribution_ratios()
        cumulative_contribution_ratios = self.get_cumulative_contribution_ratios()
        cont_cumcont_ratios = pd.concat([contribution_ratios, cumulative_contribution_ratios], axis=1).T
        cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']  # 行の名前を変更
        # 寄与率を棒グラフで、累積寄与率を線で入れたプロット図を重ねて描画
        x_axis = range(1, contribution_ratios.shape[0] + 1)  # 1 から成分数までの整数が x 軸の値
        plt.rcParams['font.size'] = 18
        plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center')  # 寄与率の棒グラフ
        plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-')  # 累積寄与率の線を入れたプロット図
        plt.xlabel('Number of principal components')  # 横軸の名前
        plt.ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')  # 縦軸の名前。\n で改行しています
        plt.show()


    def show_band_gap(self):
        # 第 1 主成分と第 2 主成分の散布図 (band_gap の値でサンプルに色付け)
        score = self.get_score()
        plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=self.data.iloc[:, 0], cmap=plt.get_cmap('jet'))
        clb = plt.colorbar()
        clb.set_label('band_gap', labelpad=-20, y=1.1, rotation=0)
        plt.xlabel('t1')
        plt.ylabel('t2')
        plt.show()


