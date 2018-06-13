import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
import config


def get_yyyymm_list(verbose=None):
    """
    2003년 5월부터 2018년 5월까지의 데이터를 모집단으로!!
    Creating yyyymm list
    """
    yyyymm_list = []
    for i in pd.date_range(datetime.datetime(2003,1,1), datetime.datetime(2018,12,31), freq='M'):
        yyyymm_list.append(str(i)[0:7])

    # 18개의 머시기로 나오는데  인덱스 영역에선 우선 원핫인코딩 처리 안하고 나중에 embeding시킬때 별도로 ??
    if verbose:
        print('1. yyyymmlist')
        print(yyyymm_list)

    return yyyymm_list


def get_acceptable_yyyymm_list():
    yyyymm_list = get_yyyymm_list();
    acceptable_yyyymm_list = []

    for idx_yyyymm, yyyymm in enumerate(yyyymm_list):
        if idx_yyyymm > len(yyyymm_list) - config.N_MONTH_TO_PREDICT:
            continue
        elif idx_yyyymm < config.N_TIME_WINDOW:
            continue
        else:
            acceptable_yyyymm_list.append(yyyymm)
    return acceptable_yyyymm_list


def get_sido_nm_list():
    """
    취한 데이터는 18개의 데이터인데,
    '전국', '수도권', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시',
    '울산광역시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '세종특별자치시',
    '경상북도', '경상남도' 이렇게 총 18개의 데이터이나
    1. '전국', '수도권' 제외할 예정!
    2. '세종특별자치시'도 초반에 8년정도의 데이터가 없으므로 0으로 interpolation시키고 할 지 고민중
    3. 1과2 제거하면 15개
    """
    # sido_nm_list = ['전국', '수도권', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시',
    #    '울산광역시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '세종특별자치시',
    #    '경상북도', '경상남도']
    #
    # sido_nm_list = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시',
    #                 '울산광역시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도',
    #                 '경상북도', '경상남도']


    sido_nm_list = ['서울특별시', '부산광역시', '대구광역시']

    return sido_nm_list


def get_fetures_nm_list():
    # features_nm_list = ['CD(91일물)', '가계대출', '건설기성액', '건설투자', '경제성장률', '국고채(3년)', '민간소비',
    #                     '설비투자', '세대수', '이동건수', '인구수', '인구이동', '정기예금금리', '종합주가지수',
    #                     '주택담보대출금리', '주택대출', '환율', '경기종합지수', '부동산시장_소비심리지수', '생산자물가지수', '소비자물가지수']

    features_nm_list = ['CD(91일물)', '가계대출']

    return features_nm_list

def get_sido_onehot_map(verbose=None):
    sido_nm_list = get_sido_nm_list()

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sido_nm_list)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)

    sido_onehot_encoded_map = {}
    for idx, sido_nm in enumerate(sido_nm_list):
        sido_onehot_encoded_map[sido_nm] = onehot_encoded[idx]

    if verbose:
        print('2. sido_onehot_encoded_list')
        print(sido_onehot_encoded_map)
        print(sido_onehot_encoded_map['전국'])
    return sido_onehot_encoded_map

if __name__ == '__main__':
    get_sido_onehot_map(True)
    get_yyyymm_list(True)