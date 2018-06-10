import pandas as pd

def load_data_set(percentage, sido_onehot):
    df_61 = pd.read_csv('data\\kab61.csv')
    df_71 = pd.read_csv('data\\kab71.csv')
    df_61_filtered = df_61[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_71_filtered = df_71[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_x = df_61_filtered.append(df_71_filtered)
    df_y = pd.read_csv('data\\y.txt', delimiter='\t', encoding='MS949')

    df_x = df_x.assign(YYYYMM=lambda x: x.YYYYMMDD.str[0:7])
    df_x = df_x.assign(SIDO_ONEHOT = lambda x: x.AreaName)

    yyyymm_array = pd.unique(df_x.YYYYMM)
    indexs_array = [pd.unique(df_x.sido_onehot), yyyymm_array]




if __name__ == '__main__':
    load_data_set(30)