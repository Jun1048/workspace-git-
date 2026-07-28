import os
import joblib
import numpy as np
from pandas import pivot_table, get_dummies
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder 
from . import my_stats

# scaling()에서 사용할 스케일러 이름과 클래스의 매핑
SCALERS = {
    'standard' : StandardScaler,
    'minmax'   : MinMaxScaler,
    'robust'   : RobustScaler,
    'maxabs'   : MaxAbsScaler,
}




def long2wide(df, hue, values, dropna=True):

    wide = pivot_table(data=df,
                       index=df.groupby(hue).cumcount(),
                       columns=hue, values=values, dropna=dropna)
    
    wide.columns.name = None
    wide.index.name = None

    return wide


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 다중 공선성 제거 기능 추가에 필요한 모듈 참조 및 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reduce_vif(df, columns=None, threshold=10.0, verbose=True):



    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정 : 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        targets = list(df.select_dtypes(include='number').columns)

    else:
        missing = []
        for c in columns:
            if c not in df.columns:
                missing.apeend(c)

        
        if missing:
            raise KeyError(f'df 에 존재하지 않는 컬럼입니다: {missing}')
        targets = list(columns)

    # 컬럼이름의 오름차순 정렬
    targets.sort()

    # 대상에서 제외되는 컬럼(종속변수, 명목형 등)은 그대로 보존하기 위해 컬럼 이름을 따로 기록
    keep = []
    for c in df.columns:
        if c not in targets:
            keep.append(c)



    # --- 2) 반복 제거 과정 ---
    work = df[targets].copy()              # 원본을 보존하기 위해 대상 변수만 복사본으로 작업
    step = 0                               # 반복 단계 카운터

    while True:
        vif = my_stats.compute_vif(work)
        max_vif = vif['VIF'].max()

        # 종료 조건: 가장 큰 VIF도 기준 미만이거나 남은 변수가 한개라면 종료
        if max_vif < threshold or len(work.columns) <= 1:
            print(f'\n완료! 남은 변수: {list(work.columns)}')
            print(f'최대 VIF = {max_vif:.2f}')
            break

        # 가장 VIF가 큰 변수를 찾아 제거하고 다시 반복
        worst = vif['VIF'].idxmax()
        step += 1
        if verbose:
            print(f'[{step}단계] {worst} 제거 (VIF = {max_vif:.1f})')
        
        work = work.drop(columns=[worst])



    # --- 3) 보존 대상 컬럼과 합쳐 원래 컬럼 순서를 유지해 반한 ---
    survived = []
    for c in df.columns:
        if c in keep or c in work.columns:
            survived.append(c)

    
    return df[survived]






# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 로그 변환 함수
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def log_transform(df, log_columns=None, log1p_columns=None, reflect_columns=None, verbose=True):



    # --- 1) 작업 준비 ---
    result = df.copy()                          # 원본을 보존하기 위해 복사본으로 작업
    report = []                                 # verbose 출력을 위해 컬럼별 변환 내역을 기록



    # --- 2) 우측 꼬리 컬럼 변환 (1) : log(x) ---
    # 0 이하의 값이 하나라도 있으면 -inf 나 NaN이 되므로 미리 막는다.
    if log_columns:
        for c in log_columns:
            if df[c].min() <= 0:
                raise ValueError(f"'{c}' 컬럼에 0 이하의 값이 있어 log(x)를 취할 수 없습니다. "
                                 f'(최솟값 {df[c].min():g}) log1p_columns로 넘기세요.')


            result[c] = np.log(df[c])
            report.append([c, '우측 꼬리', 'log(x)', 'exp(y)']) 



    # --- 3) 우측 꼬리 컬럼 변환 (2) : log(1+x) ---
    # 값이 0인 경우 log(0) = -inf가 되므로 log(1+x)를 사용한다.
    if log1p_columns:
        for c in log1p_columns:

            result[c] = np.log1p(df[c])
            report.append([c, '우측 꼬리', 'log(1+x)', 'exp(y)-1'])



    # --- 4) 좌측 꼬리 컬럼 변환 : 반사 후 log(1+x) ---
    # 최댓값에서 빼면 좌우가 뒤집혀 좌측 꼬리가 우측 꼬리가 되므로, 그 뒤 동일하게 로그를 취한다.
    # 값의 대소 관계가 뒤집히므로 회귀계수의 부호도 반대로 해석해야 한다.
    if reflect_columns:
        for c in reflect_columns:

            # 역변환하려면 이 최댓값이 반드시 필요하므로 verbose 출력에 함께 남긴다.
            max_value = df[c].max()

            result[c] = np.log1p(max_value - df[c])
            report.append([c, '좌측 꼬리',
                           f'log(1+{max_value:g}-x)',
                           f'{max_value:g}-(exp(y)-1)'])



    # --- 5) 변환 내역 출력 (변환식·역변환식과 왜도 변화) --- 
    if verbose:
        print(f'{"컬럼":10s}{"꼬리방향":10s}{"변환식":22s}{"역변환식":24s}{"왜도":>16s}')
        print('-' * 88)


        for c, side, func, inverse in report:
            before = skew(df[c].dropna())
            after = skew(result[c].dropna())
            change = f'{before:.2f} -> {after:.2f}'
            print(f'{c:10s}{side:10s}{func:22s}{inverse:24s}{change:>14s}')



    # --- 6) 변환이 적용된 데이터프레임 반환 ---
    return result







# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 로그 역변환 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def inverse_log_transform(df, log_columns=None, log1p_columns=None, 
                          reflect_columns=None, verbose=True):



    # --- 1) 작업 준비 ---
    result = df.copy()                      # 원본을 보존하기 위해 복사본으로 작업
    report = []                             # verbose 출력을 위해 컬럼별 역변환 내역을 기록



    # --- 2) 우측 꼬리 컬럼 역변환 (1) : exp(y) ---
    # log(x)의 역함수인 exp(y)로 되돌린다.
    if log_columns:
        for c in log_columns:

            result[c] = np.exp(df[c])
            report.append([c, '우측꼬리', 'exp(y)'])



    # --- 3) 우측 꼬리 컬럼 역변환 (2) : exp(y)-1 ---
    # log(1+x)의 역함수인 exp(y)-1로 되돌린다.
    if log1p_columns:
        for c in log1p_columns:

            result[c] = np.expm1(df[c])
            report.append([c, '우측 꼬리', 'exp(y)-1'])



    # --- 4) 좌측 꼬리 컬럼 역변환: 최댓값 - (exp(y)-1) ---
    # 로그를 먼저 풀고 (exp(y)-1), 그 결과를 최댓값에서 빼서 뒤집힌 대소 관계를 되돌린다.
    if reflect_columns:
        for c, max_value in reflect_columns.items():

            result[c] = max_value - np.expm1(df[c])
            report.append([c, '좌측 꼬리', f'{max_value:g}-(exp(y)-1)'])



    # --- 5) 역변환 내역 출력(역변환 식과 값의 범위 변화) ---
    if verbose:
        print(f'{"컬럼":10s}{"꼬리방향":10s}{"역변환식":24s}{"값의 범위":>28s}')
        print('-' * 76)


        for c, side, inverse in report:
            before = f'{df[c].min():.2f}~{df[c].max():.2f}'
            after = f'{result[c].min():.2f}~{result[c].max():.2f}'
            change = f'{before} -> {after}'
            print(f'{c:10s}{side:10s}{inverse:24s}{change:>26s}')



    # ---6 ) 역변환이 적용된 데이터프레임 반환 ---
    return result








# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 이상치 정제 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def replace_outlier(df, columns=None, method='bound', value=None, verbose=True):



    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정 : 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        columns = list(df.select_dtypes(include='number').columns)



    # --- 2) 작업 준비 ---
    result = df.copy()          # 원본을 보존하기 위해 복사본으로 작업
    name = method.lower().strip()

    report = []                 # verbose 출력을 위해 컬럼별 대체 내역을 기록



    # --- 3) 컬럼별 이상치 판단 및 대체 ---
    for c in columns:

        # 3-1) IQR 기반 이상치 경계 계산
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr


        # 3-2) 경계를 벗어난 값의 위치
        is_outlier = (df[c] < lower) | (df[c] > upper)
        count = is_outlier.sum()


        # 3-3) method에 따라 대체값을 정하고 이상치를 바꾼다.
        if name == 'bound':

            # 하한 / 상한 바깥의 값을 각각 하한 / 상한으로 잘라낸다.
            result[c] = df[c].clip(lower, upper)
            replaced = f'{lower:.2f} 또는 {upper:.2f}'


        else:
            if name == 'median':

                # 이상치를 제외한 정상값만으로 대푯밗을 구해야 이상치에 오염되지 않는다.
                fill = df.loc[~is_outlier, c].median()

            elif name == 'mean':
                fill = df.loc[~is_outlier, c].mean()

            else:
                fill = value


            result.loc[is_outlier, c] = fill
            replaced = f'{fill:.2f}'

        report.append([c, lower, upper, count, replaced])



    # --- 4) 대체 내역 출력 (정상 범위·이상치 개수·대체값) ---
    if verbose:
        print(f"이상치 대체 방식: '{name}' (기준: IQR × 1.5)")
        print(f'{"컬럼":12s}{"정상 범위":>26s}{"이상치":>10s}{"대체값":>22s}')
        print('-' * 68)


        for c, lower, upper, count, replaced in report:
            bound = f'{lower:.2f} ~ {upper:.2f}'
            ratio = count / len(df) * 100
            found = f'{count}개({ratio:.1f}%)'
            print(f'{c:12s}{bound:>24s}{found:>12s}{replaced:>20s}')



    # --- 5) 이상치가 대체된 데이터프레임 반환 ---
    return result




    


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 데이터 라벨링 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def scaling(df, columns = None, method='standard', save_path=None, verbose=True):



    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정 : 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        columns = list(df.selec_dtypes(include='number').columns)



    # --- 2) 작업 준비 및 스케일러 이름 정규화 ---
    result = df.copy()              # 원본을 보존하기 위해 복사본으로 작업


    # 대소문자와 뒤에 붙은 'scaler'를 떼어내 이름을 통일한다. ('StandardScaler' -> 'standard')
    name = method.lower().replace('scaler', '').strip()


    # 오타를 냈을 때 KeyError 대신 사용 가능한 이름을 알려준다.
    if name not in SCALERS:
        raise ValueError(f"지원하지 않는 스케일러입니다: '{method}' "
                         f"(사용 가능: {list(SCALERS.keys())})")   



    # --- 3) 스케일러 학습 및 변환 ---
    # 이름에 해당하는 클래스로 스케일러를 만들어 대상 컬럼의 기준값(평균, 표준편차 등)을 학습시킨다.
    scaler = SCALERS[name]()
    result[columns] = scaler.fit_transform(df[columns])



    # --- 4) 변환 내역 출력 (컬럼별 값의 범위 변화) ---
    if verbose:
        print(f'{type(scaler).__name__} 적용 ({len(columns)}개 컬럼)')
        print(f'{"컬럼":12s}{"변환 전":>22s}{"변환 후":>22s}')
        print('-' * 56)


        for c in columns:
            before = f'{df[c].min():.2f} ~ {df[c].max():.2f}'
            after = f'{result[c].min():.2f} ~ {result[c].max():.2f}'
            print(f'{c:12s}{before:>20s}{after:>22s}')



    # --- 5) 학습된 스케일러 저장 (선택) ---
    # 스케일러 안에는 train 에서 구한 기준값(평균·표준편차 등)이 들어 있으므로,
    # 이 파일이 있어야 test 데이터에 똑같은 기준을 적용할 수 있다.
    if save_path:
        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok = True)            # 경로에 없는 폴더가 있으면 만들어 준다


        joblib.dump(scaler, save_path)

        if verbose:
            print(f'\n스케일러 저장: {save_path} ({type(scaler).__name__})')



    # --- 6) 스케일링된 데이터프레임 반환 ---
    return result







# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 데이터 라벨링 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def labeling(df, columns, save_path=None, verbose=True):



    # --- 1) 작업 준비 ---
    result = df.copy()          # 원본을 보존하기 위해 복사본으로 작업
    encoders = {}               # 역변환과 test 데이터 적용을 위해 컬럼별 인코더를 보관



    # --- 2) 컬럼별 라벨 인코딩 (문자열 -> 0부터 시작하는 정수) ---
    for c in columns:
        encoder = LabelEncoder()
        result[c] = encoder.fit_transform(df[c])
        encoders[c] = encoder



    # --- 3) 변환 내역 출력 (원래 값 -> 부여된 정수) ---
    if verbose:
        for c in columns:

            # classes_ 의 순서가 곧 부여된 정수값이므로 짝지어 출력한다.
            pairs = []
            for i, v in enumerate(encoders[c].classes_):
                pairs.append(f'{v}={i}')

            print(f'{c} ({len(pairs)}종): {",".join(pairs)}')



    # --- 4) 학습된 인코더 저장(선택) ---
    # 컬럼마다 인코더가 따로 있으므로 dict 통째로 하나의 pkl 파일에 저장한다.
    if save_path:
        folder = os.path.dirname(save_path)

        if folder:

            # 경로에 없는 폴더가 있으면 만들어준다.
            os.makedirs(folder, exist_ok=True)

        joblib.dump(encoders, save_path)

        if verbose:
            print(f'\n인코더 저장: {save_path} ({len(encoders)}개 컬럼)')



    # --- 5) 라벨링이 적용된 데이터프레임 반환 ---
    return result






# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 더미변수 생성 함수 정의
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def dummies(df, columns, drop_first=True, verbose=True):



    # --- 1) 처리 대상 컬럼 결정 ---
    # 값의 종류가 2개인 컬럼은 변환 대상에서 제외한다.
    targets = []
    skipped = []

    for c in columns:
        if df[c].nunique() == 2:
            skipped.append(c)

        else:
            targets.append(c)



    # --- 2) 더미 변환 수행 ---
    # dtype = int를 지정해 True/False가 아닌 0/1로 만든다
    result = get_dummies(df, columns=targets, drop_first=drop_first, dtype=int)



    # --- 3) 변환 내역 출력(생성된 더미 컬럼과 생략된 컬럼) --- 
    if verbose:
        for co in targets:

            # 원본에 없고 결과에만 있으면서 'c_'로 시작하는 컬럼이 c로부터 생성된 더미다
            created = []
            for new in result.columns:
                if new not in df.columns and new.startswith(f'{c}_'):
                    created.append(new)


            # drop_first로 빠진 기준 범주가 무엇인지 함께 알려준다.
            dropped = ''
            if drop_first:
                dropped = f' (기준: {sorted(df[c].unique())[0]}제외)'


            print(f'{c} ({df[c].nunique()}종) -> {len(created)}개: {",".join(created)}{dropped}')


        for c in skipped:
            print(f'{c} (2종) -> 생략: 이진 변수이므로 원래 컬럼을 유지')

        print(f'\n컬럼 수: {df.shape[1]}개 -> {result.shape[1]}개')



    # --- 4) 더미 변환이 적용된 데이터프레임 반환
    return result