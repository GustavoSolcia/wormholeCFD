# modifiedNLM

A modified Non-Local Means algorithm for Rician noise based on scikit-image code.

## Usage

You can see how to import and use the filter in the following example. For a complete example: https://github.com/CIERMag-FFPaivaStudents/CFD/blob/master/ImageProcessing/denoising.py

```
from modifiedNLM.estimate.noise_estimate import rician_estimate
from modifiedNLM.filter.modified_nl_means import rician_denoise_nl_means

ricianSigma = rician_estimate(imageData)
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False,
                preserve_range=True)
    denoisedData = rician_denoise_nl_means(imageData, h=1.15 * ricianSigma, fast_mode=False,
                           **patch_kw)
```

## Installation

You should be able to build and compile the cython code with

```
python3 setup.py build_ext --inplace
```

## Licence

The majority of this code is based on the <a href="https://github.com/scikit-image/scikit-image">scikit-image</a> Non-Local Means. Then, as an redistribution with modifications, we acknowledge their <a href= "https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt">LICENSE</a>. Other parts of the code that are not related with the scikit-image can be considered under a MIT License.

## Authors

`modifiedNLM` was written by `Gustavo Solcia <gustavo.solcia@usp.br>`_.
