from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
from src import transformers

sys.modules["__main__"] = transformers

app = FastAPI()


class Client(BaseModel):
    id: int | None = None
    rn: int | None = None
    pre_since_opened: int | None = None
    pre_since_confirmed: int | None = None
    pre_pterm: int | None = None
    pre_fterm: int | None = None
    pre_till_pclose: int | None = None
    pre_till_fclose: int | None = None
    pre_loans_credit_limit: int | None = None
    pre_loans_next_pay_summ: int | None = None
    pre_loans_outstanding: int | None = None
    pre_loans_total_overdue: int | None = None
    pre_loans_max_overdue_sum: int | None = None
    pre_loans_credit_cost_rate: int | None = None
    pre_loans5: int | None = None
    pre_loans530: int | None = None
    pre_loans3060: int | None = None
    pre_loans6090: int | None = None
    pre_loans90: int | None = None
    is_zero_loans5: int | None = None
    is_zero_loans530: int | None = None
    is_zero_loans3060: int | None = None
    is_zero_loans6090: int | None = None
    is_zero_loans90: int | None = None
    pre_util: int | None = None
    pre_over2limit: int | None = None
    pre_maxover2limit: int | None = None
    is_zero_util: int | None = None
    is_zero_over2limit: int | None = None
    is_zero_maxover2limit: int | None = None
    enc_paym_0: int | None = None
    enc_paym_1: int | None = None
    enc_paym_2: int | None = None
    enc_paym_3: int | None = None
    enc_paym_4: int | None = None
    enc_paym_5: int | None = None
    enc_paym_6: int | None = None
    enc_paym_7: int | None = None
    enc_paym_8: int | None = None
    enc_paym_9: int | None = None
    enc_paym_10: int | None = None
    enc_paym_11: int | None = None
    enc_paym_12: int | None = None
    enc_paym_13: int | None = None
    enc_paym_14: int | None = None
    enc_paym_15: int | None = None
    enc_paym_16: int | None = None
    enc_paym_17: int | None = None
    enc_paym_18: int | None = None
    enc_paym_19: int | None = None
    enc_paym_20: int | None = None
    enc_paym_21: int | None = None
    enc_paym_22: int | None = None
    enc_paym_23: int | None = None
    enc_paym_24: int | None = None
    enc_loans_account_holder_type: int | None = None
    enc_loans_credit_status: int | None = None
    enc_loans_account_cur: int | None = None
    enc_loans_credit_type: int | None = None
    pclose_flag: int | None = None
    fclose_flag: int | None = None


pipeline = joblib.load('models/pipeline.pkl')


@app.post('/predict')
def prediction(item: Client):
    data = item.model_dump()
    df = pd.DataFrame([data])
    
    result = pipeline.predict_proba(df)[0,1]

    return {'probability': float(result)}
