"""modifiedNLM - A modified Non-Local Means algorithm for Rician noise based on scikit-image code."""

from .modified_nl_means import rician_denoise_nl_means

__version__ = '0.1.0'
__author__ = 'Gustavo Solcia <gustavo.solcia@usp.br>'
__all__ = ['rician_denoise_nl_means']
