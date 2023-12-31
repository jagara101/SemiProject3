# SemiProject3

## “불로소득과 부채가 총생활비에 미치는 영향” 분석 

## 프로젝트 수행기간

2023.08.14 ~ 2023.08.22

개선 버전: 2023.11.28~2023.12.01

## 프로젝트 목표/소개

“한국복지패널조사” 데이터를 토대로 가설 설정 후 그 가설이 맞는지 
데이터 분석을 통해 검증 진행

▼

해당 데이터를 기반으로 14개의 독립변수 및 1개의 종속변수 선정하여 가설 2가지를 선정하였고 우리가 세운 가설이 지지되는지 확인하기 위해 분석개요->데이터준비->이상치확인 및 정제->신뢰구간 확인->정규성 및 상관관계 확인->주성분 분석->다중선형회귀분석->결과 총 8단계로 나누어서 진행


## 담당 역할

참고 자료 수집, 데이터 분석용 Code Frame 구축, PPT Deco 총괄

개선 버전: 1인 진행, 코드 수정, 오탈자 수정, 오입력된 사진 교체, ppt 개선 등 전체적으로 수정 진행 

## Team_mate
- https://github.com/jaekim3220
- https://github.com/hajinseok11
  
### 진행 방법

1. 데이터 자료 수집 : 데이터 분석의 첫 단계로, 분석할 데이터를 수집. 이 단계에서 데이터의 출처와 형식을 정확히 확인

2. 수집자료 불러오기 : 수집한 데이터를 적절한 형식으로 불러와서 사용 가능한 형태로 가공

3. 데이터프레임 타입 확인 : 데이터프레임을 불러온 후, 데이터의 구조와 타입을 확인

4. 결측치 확인 : 데이터프레임에서 결측치(missing values)가 있는지 확인

5. 결측치 있으면 처리 : 결측치가 있을 경우, 적절한 방법으로 대체하거나 제거

6. 기초통계량 확인 : 데이터의 대략적인 분포와 통계적 특성을 파악하기 위해 기초통계량을 확인. 평균, 중앙값, 표준편차 등을 확인

7. 레그플롯, 박스플롯, 히스토그램 : 해당 그래프들을 이용하여 이상치 있는지 확인 및 도수분포 확인

8. 이상치 제거 : 이상치가 발견되면 query함수를 이용하여 제외시킴

9. 레그플롯, 히스토그램(도수 분포 제대로 됬는지 확인) 재확인 : 이상치 제거 후 분석 진행 전 더블 체크하여 변화가 있는지 확인

10. 신뢰구간 확인 : 데이터의 통계적 신뢰도를 파악하기 위해 신뢰구간을 확인. 이는 결과의 신뢰성을 판단하는데 도움을 줌

11. 각 요인별 상관관계 확인 : 산점도 행렬 및 히트맵 그래프를 이용하여 각 요인들간에 상관관계 있는 지 파악

12. 표준화 : pca분석 전, 분석에 사용할 변수를 표준화하여 변수 간 스케일 차이를 해소하고 분석 결과를 더욱 해석하기 쉽게 만듦

13. PCA 주성분분석 : PCA(주성분분석)를 사용하여 변수 간 상관관계를 파악하고 종속변수에 영향이 미미한 변수를 선정
- 차원 축소를 통해 데이터의 변수 수를 줄임

14. 데이터 분석: 회귀분석을 사용하여 데이터 간 관계를 파악하고 해석

15. 정규분포가정(정규성 검정 대체) : 정규성 가정을 확인하거나 대체 방법을 사용하여 정규성을 검토

16. 그래프,차트 제시 : 분석 결과를 시각화하여 보다 명확하게 전달하고, 그래프와 차트를 통해 패턴과 관계를 시각적으로 확인

17. 결론 제시 및 ppt제작 : 분석 결과를 바탕으로 결론을 도출하고 결과를 제시



#### p.s 

- 3명이 1조로 구성되어 처음으로 제대로 된 team project 진행

- 본 결과물은 강사님한테 평가를 받고 개선해야할 내용까지 추가 및 수정하여 만든 작업물

- 최종 개선 버전은 혼자서 전체적으로 수정해서 upload하였음
