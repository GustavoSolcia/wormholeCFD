# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Wrapper of N4 bias field correction with shrinking from SimpleITK.

"""

import os
import SimpleITK as sitk

def shrinkBiasCorrection(inputImage):

    """Bias field correction with shrinking operation.

    Parameters
    -----------
    inputImage: sitkImage
        We expect an sitkImage from sitk.ReadImage.

    Returns
    --------
    dataWithoutBias: sitkImage
        Reconstructed data (not shrinked) without bias.
    bias: sitkImage
        The removed bias field (we recommend saving this array for further analysis).

    """

    shrinkFactor = 4 # Must be a integer factor

    # Using sitk.Shrink reduces the processing time and gives good results
    shrinkedImage = sitk.Shrink(inputImage, [shrinkFactor]*inputImage.GetDimension())
    
    biasFilter = sitk.N4BiasFieldCorrectionImageFilter()
    shrinkedImageWithoutBias = biasFilter.Execute(sitk.Cast(shrinkedImage, sitk.sitkFloat32))
    bias = biasFilter.GetLogBiasFieldAsImage(inputImage)
    
    dataWithoutBias = sitk.Cast(sitk.Cast(inputImage, sitk.sitkFloat32)/sitk.Exp(bias),
                        sitk.sitkInt16)

    return dataWithoutBias, bias


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

if __name__=='__main__':

    path = os.path.abspath("/wormholeCFD/ImageProcessing")
    inputName = '/testSample/testSample.nii.gz'
    outputName = '/testSample/testSample_unbiased.nii.gz' #The forward slash is necessary to path+*Name work!
    biasName = '/testSample/testSample_biasField.nii.gz'
    
    image = sitk.ReadImage(path+inputName)
    
    dataWithoutBias, bias = shrinkBiasCorrection(image)

    writeImage(path+outputName, dataWithoutBias)
    writeImage(path+biasName, bias)
