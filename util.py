import pandas as pd
import preprocessing
import config
import numpy as np
from sklearn.model_selection import train_test_split

m = preprocessing.get_sido_onehot_map()
yyyymm_list = preprocessing.get_yyyymm_list()
sido_nm_list = preprocessing.get_sido_nm_list()
features_nm_list = preprocessing.get_fetures_nm_list()

def load_data_set(percentage, random_state):
    df_61 = pd.read_csv('data\\kab61.csv')
    df_71 = pd.read_csv('data\\kab71.csv')
    df_61_filtered = df_61[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_71_filtered = df_71[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
    df_x = df_61_filtered.append(df_71_filtered)
    df_x = df_x.assign(YYYYMM=lambda x: x.YYYYMMDD.str[0:7])
    df_x = df_x[['AreaName', 'YYYYMM', 'InfoType2', 'Values']]
    df_y = pd.read_csv('data\\y.txt', delimiter='\t', encoding='MS949')

    l_training_feature = np.zeros([len(yyyymm_list)-config.N_TIME_WINDOW-config.N_TIME_WINDOW, config.N_FEATURES, config.N_TIME_WINDOW])
    l_training_label = np.zeros([len(yyyymm_list)-config.N_TIME_WINDOW-config.N_TIME_WINDOW])

    idx_yyyymm_inserting = 0
    for idx_yyyymm, yyyymm in enumerate(yyyymm_list):
        for idx_sido_nm, sido_nm in enumerate(sido_nm_list):

            if idx_yyyymm > len(yyyymm_list) - config.N_MONTH_TO_PREDICT:
                continue
            elif idx_yyyymm < config.N_TIME_WINDOW:
                continue
            else:
                # present_idx_yyyymm = idx_yyyymm
                # present_yyyymm = yyyymm

                # for idx in range(idx_yyyymm - config.N_TIME_WINDOW,idx_yyyymm):
                for idx in range(config.N_TIME_WINDOW):
                    # create each row per feature during N_TIME_WINDOW
                    for idx_feature_nm, feature_nm in enumerate(features_nm_list):
                        v = df_x[  (df_x['AreaName'] == sido_nm_list[idx_sido_nm]) &   (df_x['YYYYMM'] == yyyymm_list[idx_yyyymm+idx]) & (df_x['InfoType2'] == features_nm_list[idx_feature_nm])]
                        if v.shape[0] != 0:
                            # assert v.values[0,1] == yyyymm_list[idx]
                            # assert v.values[0,2] == feature_nm
                            # print(v.values[0,3])
                            l_training_feature[idx_yyyymm_inserting][idx_feature_nm][idx] = v.values[0,3]

                present_rental_price_idx = df_y[(df_y['YYYYMM'] == yyyymm_list[idx])][sido_nm].values[0]
                future_rental_price_idx = df_y[(df_y['YYYYMM'] == yyyymm_list[idx + config.N_MONTH_TO_PREDICT])][sido_nm].values[0]

                gap = future_rental_price_idx - present_rental_price_idx

                label = 0
                if gap > 0 and gap >= config.PERCENTAGE_LOWER_BAND_THRESHOLD:
                    label = 1  # up
                elif gap < 0 and np.abs(gap) >= config.PERCENTAGE_LOWER_BAND_THRESHOLD:
                    label = -1  # down
                else:
                    label = 0  # no change

                l_training_label[idx_yyyymm_inserting] = label
                idx_yyyymm_inserting = idx_yyyymm_inserting + 1
    """
    it needs to be adapted multi index
    https://pandas.pydata.org/pandas-docs/version/0.22.0/advanced.html
    """
    return train_test_split(l_training_feature, l_training_label, test_size=percentage, random_state=random_state)

if __name__ == '__main__':
    df = load_data_set(30, 33)
    # print(df)