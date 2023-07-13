# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Standard deviation estimate inspired on the paper: Residual variance and residual pattern in nonlinear regression BY THEO GASSER.

"""


import numpy as np

def rician_estimate(img):
    """Estimate of standard deviation for Rician noise recommended by Wiest-Daesslé et al.
    
    Parameters
    ----------
    img: array
        Input nd array from image that you want to estimate the standard deviation.

    Return
    ------
    sigma_est: float
        Estimated standard deviation inspired on Theo Gasser equations.

    References
    ----------
        Wiest-Daesslé N, Prima S, Coupé P, Morrissey SP, Barillot C. Rician noise removal 
        by non-Local Means filtering for low signal-to-noise ratio MRI: applications to 
        DT-MRI. Med Image Comput Comput Assist Interv. 2008;11(Pt 2):171-9.
        doi: 10.1007/978-3-540-85990-1_21. PMID: 18982603; PMCID: PMC2665702.

    """

    X = img.reshape(-1) 
    t=np.arange(0,len(X),1) 

    size=len(X)-2
    sigma=0
    for i in range(1,size):
        a = (t[i+1] - t[i]) / (t[i+1] - t[i-1])
        b = (t[i] - t[i-1]) / (t[i+1] - t[i-1])
        c = 1 / (a * a + b * b + 1)        
        
        sigma += c * c * (a * X[i-1] + b * X[i+1] - X[i]) * (a * X[i-1] + b * X[i+1] - X[i])
    sigma_est = np.sqrt(sigma / size)
    return sigma_est
