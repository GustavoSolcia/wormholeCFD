import numpy as np
cimport numpy as cnp
from _shared.fused_numerics cimport np_floats
from _shared.fast_exp cimport _fast_exp
cnp.import_array()

def fast_exp(np_floats x):
    return _fast_exp(x)

