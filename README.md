# 정형데이터 기반 아파트 전세/매매 가격 예측

## Intro
기존의 ARIMA등 통계 기반 모델을 가지고 하는게 아니라 Conditional CNN 적용해보았으며, 통계청등 다양한 정부에서 제공되는 데이터 및 크롤링된 데이터를 가지고 다양한 정형 데이터를 시도별 월별로 구성하여 이 데이터를 이용해서 향후 6개월 이후의 전세가격, 매매가격(현재 아파트만) 예측하는게 목표

## Sources
소스는 크게 4가지이고 여기서 사용할 정형데이터는 data 폴더에 위치해있습니다.

main.py : 모델 생성 및 빌드 그리고 훈련까지 담당

config.py : 아래 파라메터 설정 

preprocessing.py : 시도별 월별 기준 데이터 생성

util.py : training, test 데이터셋 로드

### How to run


### Temporal CNN
```
run main.py
```

### Conditional CNN
```
run main-conditional-cnn.py
```

## Architecture 

### Model #1
![model1](/materials/model1.jpg)


### Model #2
![model2](/materials/model2.jpg)


### Parameters
you can change parameters editing config.py 
```
N_FEATURES = 피처의 개수

N_TIME_WINDOW = 타임 윈도우 크기

N_MONTH_TO_PREDICT = 예측할 월 -  현시점 

PERCENTAGE_UPPER_BAND_THRESHOLD = 상승 Upper band 기준

PERCENTAGE_LOWER_BAND_THRESHOLD = 상승 Lower band 기준 

PICKLE_FILE_NAME = PICKLE 파일 명
```

## Plan
TBD

## Reference
1. [IMF이후아파트전세가율에관한연구](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=f9976b189dbbf0dbffe0bdc3ef48d419)

2. [아파트매매가격과전세가격의상호확산효과](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=916352bebbead0807ecd42904f0c5d65)

3. [거시경제변수를고려한서울아파트가격의추세-순환분석](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=5752bb743fe5455bd18150b21a227875#redirect)

4. [APT 전세가와매매가상호영향력에관한실증적연구: 강남지역APT를중심으로](http://www.riss.kr/search/detail/)
