import pandas as pd
import numpy as np


def remove_outliers_iqr(df_in, df_out):  # outlier 제거 함수 정의

    combined_df = pd.concat([df_in, df_out],
                            axis=1)  # pd.concat = 2개를 합치기 // 입력 변수(X)와 출력 변수(y)를 합쳐서 전체 데이터프레임 생성 // x랑 y를 한번에 고려
    # axis = 1 --- 오른쪽으로 합치기 --- 데이터 구조를 보면 예쁘게 정리가 된다

    numeric_cols = combined_df.select_dtypes(
        include=np.number).columns  # .select_dtypes 를 통해 특정 열만 출력 outlier 를 탐지할 숫자형 컬럼만 선택
    # numeric_cols = 숫자로만 구성된 열들의 이름 리스트 (.colums = 그 열의 이름을 리스트로 출력)

    # 각 컬럼에 대해 이상치 경계 계산
    Q1 = combined_df[numeric_cols].quantile(0.25)  # .quantile(0.25) = 데이터를 오름차순으로 정렬했을 떄 하위 25% 지점
    Q3 = combined_df[numeric_cols].quantile(0.75)  # .quantile(0.75) = 데이터를 오름차순으로 정렬했을 떄 상위 25% 지점
    IQR = Q3 - Q1  # IQR은 대략 상위 25% - 상위75% = 중간정도의 값에 해당

    lower_bound = Q1 - 6 * IQR  # 보통은 1.5* IQR을 진행하지만 최대한 삭제되는 데이터가 적도록 진행
    upper_bound = Q3 + 6 * IQR

    # 밑의 줄은 공부를 더 해보자
    # 모든 컬럼에 대해 정상 범위 내에 있는 데이터만 True로 표시
    # (row의 어떤 컬럼이라도 이상치면 해당 row 전체가 False가 됨)
    condition = ~((combined_df[numeric_cols] < lower_bound) | (combined_df[numeric_cols] > upper_bound)).any(axis=1)

    # 정상 범위에 있는 데이터만 필터링
    df_in_no_outliers = df_in[condition]
    df_out_no_outliers = df_out[condition]

    return df_in_no_outliers, df_out_no_outliers
