"""
통계분석 유틸리티
@Author: 이광호(leekh4232@gmail.com)
"""
import sys
import numpy as np
import seaborn as sb
from pca import pca
from math import sqrt
from tabulate import tabulate
from matplotlib import pyplot as plt

from pandas import DataFrame, MultiIndex, Series

from scipy import stats
from scipy.stats import t, pearsonr, spearmanr

from statsmodels.formula.api import ols, logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler



def prettyPrint(df, headers="keys", tablefmt="psql", numalign="right", title="value"):

    if isinstance(df, Series):
        df=DataFrame(df,columns=[title])
        
    print(tabulate(df, headers=headers, tablefmt=tablefmt, numalign=numalign))


def getConfidenceInterval(data, clevel=0.95, isPrint=True):
    """
    신뢰구간 계산

    Parameters
    -------
    - data: 데이터
    - clevel: 신뢰수준. 기본값은 0.95

    Returns
    -------
    - cmin: 신뢰구간 하한
    - cmax: 신뢰구간 상한
    """
    n = len(data)                           # 샘플 사이즈
    dof = n - 1                             # 자유도
    sample_mean = data.mean()               # 표본 평균
    sample_std = data.std(ddof=1)           # 표본 표준 편차
    sample_std_error = sample_std / sqrt(n)  # 표본 표준오차

    # 신뢰구간
    cmin, cmax = t.interval(
        clevel, dof, loc=sample_mean, scale=sample_std_error)

    if isPrint:
        df = DataFrame({
            "신뢰구간": [cmin, cmax]
        }, index=['하한', '상한'])

        prettyPrint(df)
    else:
        return (cmin, cmax)


def pearson_r(df, isPrint=True):
    """
    피어슨 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = pearsonr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    if isPrint:
        prettyPrint(rdf)
    else:
        return rdf


def spearman_r(df, isPrint=True):
    """
    스피어만 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = spearmanr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    if isPrint:
        prettyPrint(rdf)
    else:
        return rdf


class OlsResult:
    def __init__(self):
        self._x_train = None
        self._y_train = None
        self._train_pred = None
        self._x_test = None
        self._y_test = None
        self._test_pred = None
        self._model = None
        self._fit = None
        self._summary = None
        self._table = None
        self._result = None
        self._goodness = None
        self._varstr = None
        self._coef = None
        self._intercept = None
        self._trainRegMetric = None
        self._testRegMetric = None
    
    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def train_pred(self):
        return self._train_pred

    @train_pred.setter
    def train_pred(self, value):
        self._train_pred = value

    @property
    def x_test(self):
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        self._x_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def test_pred(self):
        return self._test_pred

    @test_pred.setter
    def test_pred(self, value):
        self._test_pred = value

    @property
    def model(self):
        """
        분석모델
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        """
        분석결과 객체
        """
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        """
        분석결과 요약 보고
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def table(self):
        """
        결과표
        """
        return self._table

    @table.setter
    def table(self, value):
        self._table = value

    @property
    def result(self):
        """
        결과표 부가 설명
        """
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def goodness(self):
        """
        모형 적합도 보고
        """
        return self._goodness

    @goodness.setter
    def goodness(self, value):
        self._goodness = value

    @property
    def varstr(self):
        """
        독립변수 보고
        """
        return self._varstr

    @varstr.setter
    def varstr(self, value):
        self._varstr = value
    
    @property
    def coef(self):
        return self._coef

    @coef.setter
    def coef(self, value):
        self._coef = value

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        self._intercept = value

    @property
    def trainRegMetric(self):
        return self._trainRegMetric

    @trainRegMetric.setter
    def trainRegMetric(self, value):
        self._trainRegMetric = value

    @property
    def testRegMetric(self):
        return self._testRegMetric

    @testRegMetric.setter
    def testRegMetric(self, value):
        self._testRegMetric = value

    def setRegMetric(self, y_train, y_train_pred, y_test=None, y_test_pred=None):
        self.trainRegMetric = RegMetric(y_train, y_train_pred)

        if y_test is not None and y_test_pred is not None:
            self.testRegMetric = RegMetric(y_test, y_test_pred)


def myOls(data, y=None, x=None, expr=None):
    """
    회귀분석을 수행한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # 데이터프레임 복사
    df=data.copy()

    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    if not expr:
        #독립변수의 이름이 리스트가 아니라면 리스트로 변환
        if type(x) != list:
            x=[x]
        expr="%s~%s" % (y, "+".join(x))
    
    else:
        x = []
        p = expr.find('~')
        y = expr[:p].strip()
        x_tmp = expr[p+1:]
        x_list = x_tmp.split('+')

        for i in x_list:
            k=i.strip()

            if k:
                x.append(k)

    # 회귀모델 생성
    model = ols(expr, data=data)
    # 분석 수행
    fit = model.fit()

    # 파이썬 분석결과를 변수에 저장한다.
    summary = fit.summary()

    # 회귀 계수 (beta) 추출
    beta_values = fit.params.drop("Intercept")  # Intercept 변수 제외

    # 첫 번째, 세 번째 표의 내용을 딕셔너리로 분해
    my = {}

    for k in range(0, 3, 2):
        items = summary.tables[k].data
        # print(items)

        for item in items:
            # print(item)
            n = len(item)

            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i+1].strip()

                if key and value:
                    my[key] = value

    # 두 번째 표의 내용을 딕셔너리로 분해하여 my에 추가
    my['variables'] = []
    name_list = list(data.columns)
    #print(name_list)

    for i, v in enumerate(summary.tables[1].data):
        if i == 0:
            continue

        # 변수의 이름
        name = v[0].strip()

        vif = 0

        # Intercept는 제외
        if name in name_list:
            # 변수의 이름 목록에서 현재 변수가 몇 번째 항목인지 찾기 
            j = name_list.index(name)
            vif = variance_inflation_factor(data, j)

        my['variables'].append({
            "name": name,
            "coef": v[1].strip(),
            "std err": v[2].strip(),
            "t": v[3].strip(),
            "P-value": v[4].strip(),
            "Beta": beta_values.get(name, 0),
            "VIF": vif,
        })

    # 결과표를 데이터프레임으로 구성
    mylist = []
    yname_list = []
    xname_list = []

    for i in my['variables']:
        if i['name'] == 'Intercept':
            continue

        yname_list.append(y)
        xname_list.append(i['name'])

        item = {
            "B": i['coef'],
            "표준오차": i['std err'],
            "β": i['Beta'],
            "t": "%s*" % i['t'],
            "유의확률": i['P-value'],
            "VIF": i["VIF"]
        }

        mylist.append(item)

    table = DataFrame(mylist,
                   index=MultiIndex.from_arrays([yname_list, xname_list], names=['종속변수', '독립변수']))
    
    # 분석결과
    result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

    # 모형 적합도 보고
    goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

    # 독립변수 보고
    varstr = []

    for i, v in enumerate(my['variables']):
        if i == 0:
            continue
        
        s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
        k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y, '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

        varstr.append(k)

    ols_result = OlsResult()
    ols_result.model = model
    ols_result.fit = fit
    ols_result.summary = summary
    ols_result.table = table
    ols_result.result = result
    ols_result.goodness = goodness
    ols_result.varstr = varstr

    return ols_result


def scalling(df, yname=None):
    """
    데이터 프레임을 표준화 한다.

    Parameters
    -------
    - df: 데이터 프레임
    - yname: 종속변수 이름

    Returns
    -------
    - x_train_std_df: 표준화된 독립변수 데이터 프레임
    - y_train_std_df: 표준화된 종속변수 데이터 프레임
    """
    # 평소에는 yname을 제거한 항목을 사용
    # yname이 있지 않다면 df를 복사
    x_train = df.drop([yname], axis=1) if yname else df.copy()
    x_train_std = StandardScaler().fit_transform(x_train)
    x_train_std_df = DataFrame(x_train_std, columns=x_train.columns)
    
    if yname:
        y_train = df.filter([yname])
        y_train_std = StandardScaler().fit_transform(y_train)
        y_train_std_df = DataFrame(y_train_std, columns=y_train.columns)

    if yname:
        result = (x_train_std_df, y_train_std_df)
    else:
        result = x_train_std_df

    return result


class LogitResult:
    def __init__(self):
        self._model = None    
        self._fit = None
        self._summary = None
        self._prs = None
        self._cmdf = None
        self._result_df = None
        self._odds_rate_df = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def prs(self):
        return self._prs

    @prs.setter
    def prs(self, value):
        self._prs = value

    @property
    def cmdf(self):
        return self._cmdf

    @cmdf.setter
    def cmdf(self, value):
        self._cmdf = value

    @property
    def result_df(self):
        return self._result_df

    @result_df.setter
    def result_df(self, value):
        self._result_df = value

    @property
    def odds_rate_df(self):
        return self._odds_rate_df

    @odds_rate_df.setter
    def odds_rate_df(self, value):
        self._odds_rate_df = value

