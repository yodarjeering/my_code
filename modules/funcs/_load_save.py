import pandas as pd
import pickle


def load_csv(load_path):
    df = pd.read_csv(load_path, index_col=0)
    return df

def save_pickle(save_path,object_):
    with open(save_path, mode="wb") as f:
        pickle.dump(object_, f)
        
def load_pickle(save_path):
    with open(save_path, mode="rb") as f:
        object_ = pickle.load(f)
    return object_