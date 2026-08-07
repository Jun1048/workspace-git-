import numpy as np
from pandas import DataFrame, Series, Grouper, concat, to_datetime


# 정상성 검정·자기상관 계수 계산
from statsmodels.tsa.stattools import adfuller


# 시계열 분해
from statsmodels.tsa.seasonal import seasonal_decompose





# --------------------------
# 시계열 인덱스 설정 함수 정의
# --------------------------
def set_index(data, column, freq=None, sort=True):

    """ 날짜 컬럼을 인덱스로 지정하고 관측 간격을 명시한다.

    Args:
        data (DataFrame): 날짜 컬럼을 포함하는 데이터프레임.
        column (str): 인덱스로 사용할 날짜 컬럼명.
        freq (str): 관측 간격 (예. "MS", "D", "W"). None이면 지정하지 않음(기본값:None).
        sort (bool): 인덱스를 오름차순으로 정렬할지 여부 (기본값: True).
    
    
    Returns:
        DataFrame: 날짜 인덱스가 설정된 데이터프레임.
    
    """

    df = data.copy()



    # --- 1) 날짜 컬럼이 문자열이면 datetime으로 변환 ---
    if df[column].dtype == "object":
        df[column] = to_datetime(df[column])



    # --- 2) 인덱스로 지정 ---
    df = df.set_index(column)

    if sort:
        df = df.sort_index()



    # --- 3) 관측 간격 명시 ---
    # 간격을 명시해야 window·period 값이 "시간"을 뜻하게 된다.
    # 간격이 없으면 12는 "12개월"이 아니라 그냥 "관측치 12개"일 뿐이다.
    if freq is not None:
        df = df.asfreq(freq)

    return df





# -------------------------------------------------
# 시계열 데이터를 구간으로 나누어 구간별 통계를 내는 함수
# -------------------------------------------------
def report_split(data, column=None, size=3):

    """ 시계열을 여러 구간으로 잘라 구간별 평균·표준편차·변동계수를 비교한다.

    Args:
        data (DataFrame) | Seires): 날짜 인덱스를 가진 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        size (int): 나눌 구간의 개수 (기본값: 3).
    
    
    Returns:
        DataFrame: 구간별 시작·종료·관측치 수·평균·표준편차·변동계수.

    """

    if column is not None:              # data가 데이터프레임이면 대상 컬럼만 추출
        data = data[column]


    length = len(data) // size          # 구간 하나의 길이
    result = []                         # 결과를 담을 리스트


    for i in range(size):

        # i번째 구간을 잘라낸다
        part = data.iloc[i * length : (i + 1) * length]


        result.append({
            "구간": f"구간 {i + 1}",
            "시작": part.index[0].strftime("%Y-%m-%d"),
            "종료": part.index[-1].strftime("%Y-%m-%d"),
            "관측치 수": len(part),
            "평균": round(part.mean(), 3),
            "표준편차": round(part.std(), 3),
            "변동계수": round(part.std() / part.mean(), 4),
        }) 


    return DataFrame(result).set_index("구간")





# -----------------------
# 기간별 변동 보고 함수 정의
# -----------------------
def report_variation(data, column=None, freq="YE"):


    """ 기간 단위로 묶어 변동폭이 수준에 비례해 커지는지 확인한다.

    표준편차는 커지는데 변동계수(표준편차/평균)가 일정하면 변동폭이 수준에 비례한다는 뜻이다.
    이 경우 로그 변환이 필요하다.


    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None)/
        freq (str): 묶을 기간 단위 (기본값: "YE").
            YE → 연말 기준 1년씩 / QE → 분기씩 / ME → 월말 기준 한 달씩,
            WE → 주말 기준 1주일식 / D → 하루씩
    
            
    Returns:
        DataFrame: 기간별 평균·표준편차·변동계수.
    
    """



    # --- 0) 대상 컬럼 추출 ---
    if column is not None:
        data = data[column]



    # --- 1) 기간 단위로 묶어 평균과 표준편차를 계산 ---
    # groupyby()는 "값이 같은 것끼리" 묶는데, 날짜는 값이 전부 달라서 그대로는 묶이지 않는다.
    # Grouper(freq=...)를 키로 주면 "이 시간 간격으로 잘라서 묶어라"라는 뜻이 된다.
    # freq='YE' → 연말 기준 1년씩 / "QE" → 분기씩 / "ME" → 월말 기준 한 달씩
    # 인덱스는 각 구간의 끝 날짜가 된다. (1949년 묶음 → 1949-12-31)
    group = data.groupby(Grouper(feq=freq))


    result_df = DataFrame({
        "평균": group.mean().round(3),
        "표준편차": group.std().round(3),
    })



    # --- 2) 변동계수 = 표준편차 / 평균 ---
    result_df["변동계수"] = (result_df["표준편차"] / result_df["평균"]).round(4)
    result_df = result_df.dropna()
    result_df.index.name = "기간"



    # --- 3) 최대 / 최소 비율로 판정 ---
    std_ratio = result_df["표준편차"].max() / result_df["표준편차"].min()
    cv_ratio = result_df["변동계수"].max() / result_df["변동계수"].min()


    if cv_ratio < std_ratio:
        conclusion = "변동폭이 수준에 비례한다 → 로그변환 권장 / 승법(multiplicative) 모델"

    else:
        conclusion - "변동폭이 일정하다 → 원본 사용 가능 / 가법(additive) 모델"


    print(f"표준편차 최대/최소: {std_ratio:.2f}배")
    print(f"변동계수 최대/최소: {cv_ratio:.2f}배")
    print(f"판정: {conclusion}")


    return result_df





# -----------------
# ADF 검정 함수 정의
# -----------------
def adf_test(data, column=None, name=None, alpha=0.05):


    """ 시계열 하나의 정상ㅇ성을 ADF 검정으로 판정한다.

    여러 대상을 비교하는 adf_dff · adf_tarnsform 도 이 함수를 반복 호출한다.
    반환값이 가로 1행 표이므로 concat()으로 이어 붙이면 그대로 여러 행이 된다.


    Args:
        data (DataFrame) | Series): 검정할 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None)/
        name (str): 결과표에 표시할 행 이름 (기본값: None).
            생략하면 컬럼명 → Series의 이름 → "시계열" 순으로 찾아 쓴다.
        alpha (float): 유의수준 (기본값: 0.05)/

    
    Returns:
        DataFrame: 대상을 인덱스로 하는 가로 1행 ADF 검정 결과표.

    """


    # --- 1) 기본 준비 및 정상성 검정 ---
    if column is not None:                  # 컬럼명이 있다면
        data = data[column]                 # 대상 데이터만 추출


    # 표의 행 이름 가져오기
    if name is None:                        # 이름을 직접 주지 않았다면
        if column is not None:
            name = column                   # 컬럼명을 행 이름으로 사용

        else:
            name = data.name                # Series의 이름을 행 이름으로 사용

    if name is None:    name = "시계열"      # 그래도 이름이 없으면 그냥 "시계열"로 지정


    x = Series(data).dropna()               # 결측치 제거


    # adfuller()는 결과를 튜플로 돌려준다
    statistic, pvalue, usedlag, nobs, cvalues, icbest = adfuller(x)



    # --- 2) 결과 구성 및 반환 ---
    stationary = bool(pvalue < alpha)       # p-value가 작아야 정상성 확보


    # 결과표 반환
    return DataFrame([{
        "관측치 수": len(x),
        "검정통계량(ADF)": round(statistic, 3),
        "p-value": round(pvalue, 4),
        "사용 시차": usedlag,
        "1% 기각값": round(cvalues["1%"], 3),
        "5% 기각값": round(cvalues["5%"], 3),
        "10% 기각값": round(cvalues["10%"], 3),
        "표준편차": round(x.std(), 3),
        "정상성": stationary,
        "판정": "정상" if stationary else "비정상",
    }], index = [name])
