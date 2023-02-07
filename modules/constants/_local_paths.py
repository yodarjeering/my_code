import os
import dataclasses

@dataclasses.dataclass(frozen=True)
class LocalPaths:
    
    # プロジェクトルートの絶対パス
    BASE_DIR: str = os.path.abspath('./')
    ## dataディレクトリまでの絶対パス
    DATA_DIR: str = os.path.join(BASE_DIR,'data')
    
    ### rawディレクトリパス
    RAW_DIR: str = os.path.join(DATA_DIR,'raw')
    ### csvファイルまでのパス
    RAW_TOPIX_PATH: str = os.path.join(RAW_DIR,'TOPIX_10years.csv')
    RAW_DAW_PATH: str = os.path.join(RAW_DIR,'DAW_10years.csv')
    RAW_225_PATH: str = os.path.join(RAW_DIR,'NK225_10years.csv')
    RAW_TOPIX_LATEST_PATH:str = os.path.join(RAW_DIR,'TOPIX_latest.csv')
    RAW_DAW_LATEST_PATH : str = os.path.join(RAW_DIR,'DAW_latest.csv')
    
    
    # path_tpx = '/Users/Owner/Desktop/StockPriceData/Stock_index/TOPIX_10years.csv'
    # path_225 = '/Users/Owner/Desktop/StockPriceData/Stock_index/NK225_10years.csv'
    # path_daw = '/Users/Owner/Desktop/StockPriceData/Stock_index/DAW_10years.csv'
    # path_bear = '/Users/Owner/Desktop/StockPriceData/Stock_index/R225BEAR_10years.csv'
    # path_tpx_sim = '/Users/Owner/Desktop/StockPriceData/TOPIX/TOPIX_20211208.csv'
    # path_daw_sim = '/Users/Owner/Desktop/StockPriceData/DAW/DAW_20211208.csv'
    # save_pickle_path = '/Users/Owner/Desktop/program/Sotsuron/code/wave_pickles'