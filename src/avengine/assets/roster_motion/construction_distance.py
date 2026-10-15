"""Route proposal margin; final original QA predicates remain authoritative."""
import numpy as np

def accepts_distance_trend(values,sign,minimum=.5,tolerance=.05):
    values=np.asarray(values,float)
    if len(values)<2 or not np.isfinite(values).all():return False
    directed=float(sign)*values
    excursion=float(np.max(np.maximum.accumulate(directed)-directed))
    return directed[-1]-directed[0]>=float(minimum)-1e-10 and excursion<=float(tolerance)+1e-10
