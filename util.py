import pandas as pd
import preprocessing

m = preprocessing.get_sido_onehot_map()

def load_data_set(percentage):
    df_61 = pd.read_csv('data\\kab61.csv')
    df_71 = pd.read_csv('data\\kab71.csv')
    df_61_filtered = df_61[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_71_filtered = df_71[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_x = df_61_filtered.append(df_71_filtered)
    df_y = pd.read_csv('data\\y.txt', delimiter='\t', encoding='MS949')

    df_x = df_x.assign(YYYYMM=lambda x: x.YYYYMMDD.str[0:7])
    # df_x = df_x.assign(SIDO_ONEHOT=lambda x: m.get[x.AreaName])

    """
    it needs to be adapted multi index
    https://pandas.pydata.org/pandas-docs/version/0.22.0/advanced.html
    """

    return df_x

if __name__ == '__main__':
    df = load_data_set(30)
    print(df)