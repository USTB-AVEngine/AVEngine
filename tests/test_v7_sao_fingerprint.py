"""A short duplicate reference must survive bidirectional preselection."""
import numpy as np
import pytest
from scipy.signal import chirp
from avengine.qa.v7_auto_admission import audio_fingerprint,fingerprint_index,fingerprint_alignment
from tools.qa.admit_sao_topup import candidate_reference_keys,duplicate
pytestmark=pytest.mark.fast_unit

def test_short_embedded_reference_is_checked_even_when_long_query_fraction_is_small():
 rng=np.random.default_rng(1907)
 t=np.arange(16000)/16000
 reference=.07*chirp(t,f0=300,f1=2700,t1=1)+rng.normal(0,.004,len(t))
 bulk=rng.normal(0,.025,61440)
 query=np.concatenate([bulk,reference,bulk[::-1]])
 q=audio_fingerprint(query,16000);r=audio_fingerprint(reference,16000)
 qi=fingerprint_index(q,query=True);ri=fingerprint_index(r)
 common=len(set(qi)&set(ri))
 assert common>=12
 assert common/len(qi)<.20
 assert duplicate(fingerprint_alignment(r,q))
 assert "short" in candidate_reference_keys({"short":common},len(qi),{"short":ri})
