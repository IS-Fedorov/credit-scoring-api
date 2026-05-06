from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
import gc

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('Step 1: CreditFeatureEngineer')

        paym_cols = [c for c in X.columns if c.startswith('enc_paym_')]

        X['enc_mode'] = X[paym_cols].mode(axis=1, dropna=True).iloc[:, 0].fillna(-1).astype('Int8')
        gc.collect()
        X['bad_pay_count'] = (X[paym_cols] > 0).sum(axis=1).astype('Int8')
        gc.collect()
        X['paym_status_variance'] = X[paym_cols].var(axis=1).fillna(0).astype('float32')

        stayed_columns = [
            'id', 'rn', 'pre_since_opened', 'pre_since_confirmed', 'pre_pterm',
            'pre_till_pclose', 'pre_loans_credit_limit', 'pre_loans_next_pay_summ',
            'pre_loans_outstanding', 'pre_loans_max_overdue_sum',
            'pre_loans_credit_cost_rate', 'is_zero_loans5', 'is_zero_loans530',
            'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 'pre_util',
            'pre_over2limit', 'pre_maxover2limit', 'is_zero_util', 'is_zero_maxover2limit',
            'enc_loans_credit_status', 'enc_loans_account_cur',
            'enc_loans_credit_type', 'pclose_flag', 'enc_paym_0', 'enc_paym_5',
            'enc_paym_8', 'enc_paym_11', 'enc_mode', 'bad_pay_count',
            'paym_status_variance'
        ]

        for c in X.select_dtypes(include=['int64', 'int32']).columns:
          X[c] = pd.to_numeric(X[c], downcast='integer')

        gc.collect()

        return X[stayed_columns]
    

class DataAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, agg_dict):
        self.agg_dict = agg_dict

    def fit(self, X, y=None):
        return self

    @staticmethod
    def get_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else None

    def transform(self, X):
        print('Step 2: DataAggregator')

        group = X.groupby('id').agg(self.agg_dict)
        group.columns = [f'{c[0]}_{c[1]}' if isinstance(c, tuple) else c for c in group.columns]

        last_state = X.sort_values(['id', 'rn']).drop_duplicates('id', keep='last')

        last_state.columns = [c if c == 'id' else f'{c}_last' for c in last_state.columns]

        final_df = pd.merge(group.reset_index(), last_state, on='id', how='left')

        for c in final_df.select_dtypes(include=['float64']).columns:
            final_df[c] = final_df[c].astype('float32')

        gc.collect() # Принудительный сбор мусора

        return final_df
    

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, final_df, y=None):
        return self

    def transform(self, final_df):
        print('Step 3: ColumnDropper')

        return final_df.drop(columns=self.columns_to_drop)