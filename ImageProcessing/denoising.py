# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Denoising using a modified Non-Local Means for Rician noise.

"""

import sys

sys.path.append('../NLM') #ATENTION: This path depends on where you cloned our NLM repository

import os
import SimpleITK as sitk
from modifiedNLM.estimate.noise_estimate import rician_estimate
from modifiedNLM.filter.modified_nl_means import rician_denoise_nl_means


def NLM(imageData):

    """Wrapper of modified NLM imported from https://github.com/CIERMag-FFPaivaStudents/NLM.

    Parameters
    ----------
    imageData: array
        Numpy array from image desired to denoise. Atention: Be shure your image has Rician noise.

    Returns
    -------
    denoisedData: array
        Denoised array from rician_denoise_nl_means.

    """

    ricianSigma = rician_estimate(imageData)
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False,
                preserve_range=True)
    denoisedData = rician_denoise_nl_means(imageData, h=1.15 * ricianSigma, fast_mode=False,
                           **patch_kw)
    return denoisedData

def createSITKcopy(image, array):

    """Create new sitk image from array with same information from base image.

    Parameters
    ----------
    image: sitkImage
        Base image with the information that you want to copy.
    array: array
        Numpy array that you want to transform in sitkImage.

    Returns
    -------
    copy: sitkImage
        Transformed array with the same information from the base image.

    """
    
    copy = sitk.GetImageFromArray(array)
    copy.CopyInformation(image)

    return copy

def writeImage(dataPath, data):

    """Wrapper of image writing operation from SimpleITK.

    Parameters
    ----------
    dataPath: string
        String containing a path to the directory + the data name.
    data: sitkImage
        sitkImage that you want to save.

    """
    standard_writer = sitk.ImageFileWriter()
    standard_writer.SetFileName(dataPath)
    standard_writer.Execute(data)

if __name__ == '__main__':

    path = os.path.abspath('/wormholeCFD/ImageProcessing')
    inputName = '/testSample/testSample_unbiased.nii.gz'
    outputName = '/testSample/testSample_denoised.nii.gz'
    image = sitk.ReadImage(path+inputName)

    imageData = sitk.GetArrayViewFromImage(image)

    denoisedData = NLM(imageData)

    denoised = createSITKcopy(image, denoisedData)

    writeImage(path+outputName, denoised)
