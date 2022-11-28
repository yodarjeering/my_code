import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# 標準化する関数　平均 -> 0 分散 -> 1
def standarize(df):
    # ddof = 0 : 分散
    # ddof = 1 : 不偏分散
    df = (df - df.mean())/df.std(ddof=0)
    return df

# 標準化後の値をもとに戻す関数
def inverse(after,std_,mean_):
    after = after*std_ + mean_
    return after 

# コサイン類似度を計算する関数
def cos_sim(vec1,vec2):
    inner_product = vec1 @ vec2
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    norm_product = vec1_norm*vec2_norm
    cos = inner_product/norm_product
    return cos


