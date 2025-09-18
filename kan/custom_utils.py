import pandas as pd
import numpy as np


def remove_outliers_iqr(df_in, df_out, rr=6):

    combined_df = pd.concat([df_in, df_out],
                            axis=1)

    numeric_cols = combined_df.select_dtypes(
        include=np.number).columns

    Q1 = combined_df[numeric_cols].quantile(0.25)
    Q3 = combined_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1  # IQR은 대략 상위 25% - 상위75% = 중간정도의 값에 해당

    lower_bound = Q1 - rr * IQR  # 보통은 1.5* IQR을 진행하지만 최대한 삭제되는 데이터가 적도록 진행
    upper_bound = Q3 + rr * IQR

    condition = ~((combined_df[numeric_cols] < lower_bound) | (combined_df[numeric_cols] > upper_bound)).any(axis=1)

    df_in_no_outliers = df_in[condition]
    df_out_no_outliers = df_out[condition]

    return df_in_no_outliers, df_out_no_outliers
