# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Wrapper of atropos segmentation from advanced normalization tools (ANTs).

"""
import os 
import ants
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def createSigmoidMask(image):

    """Creates sigmoid mask using sitk filter.

    Parameters
    -----------
    image: sitkImage
        We expect an sitkImage from sitkReadImage.

    Returns
    --------
    sigmoidMask: sitkImage
        Sigmoid mask with beta value calculated with non-zero values.
        

    """

    MAX = 1
    MIN = 0
    ALPHA = 1.0

    flattenIntensityArray = sitk.GetArrayViewFromImage(image).reshape(-1)
    frequency, bins, _ = plt.hist(flattenIntensityArray, bins=256)
    beta = bins[frequency[1:-1].argmax()+1] # non-zero values
    
    sigmoidFilter = sitk.SigmoidImageFilter()
    sigmoidFilter.SetOutputMinimum(MIN)
    sigmoidFilter.SetOutputMaximum(MAX)
    sigmoidFilter.SetAlpha(ALPHA)
    sigmoidFilter.SetBeta(beta)
    
    sigmoidMask = sigmoidFilter.Execute(image)

    return sigmoidMask


def convertSitkToAnts(sitkImage):

    """Conversor of sitkImage to antsImage.

    Parameters
    -----------
    sitkImage: sitkImage
        Image desired to convert.


    Returns
    --------
    antsImage: antsImage
        Converted image.


    """

    array = sitk.GetArrayFromImage(sitkImage)
    antsImage = ants.from_numpy(array.astype('float32'))

    return antsImage

def getSegmentationArray(antsImage, antsMask):
    
    """Atropos segmentation from Advanced Normalization Tools returned as a numpy array.

    Parameters
    ----------
    antsImage: antsImage
        Image you want to apply segmentation.
    antsMask: antsImage
        Mask used on atropos segmentation.

    Returns
    -------
    segmentationArray: array
        Segmented image on array with corrected axis.

    """

    segmentationAnts = ants.atropos(a=antsImage, m='[0.1, 1x1x1]', c='[50, 0.0001]',
                                i='kmeans[3]', p='Socrates[1]', x=antsMask)

    segmentationArray = segmentationAnts['segmentation'].numpy()

    return segmentationArray

def writeArrayImage(dataPath, sitkImage, array):

    """Image writing operation with array conversion.

    Parameters
    ----------
    dataPath: string
        String containing a path to the directory + the data name.
    sitkImage: sitkImage
        sitkImage (usually the input) that we use to copy information.
    array: array
        Numpy array you want to save.


    """

    newSitkImage = sitk.GetImageFromArray(array)
    newSitkImage.CopyInformation(sitkImage)

    standard_writer = sitk.ImageFileWriter()
    standard_writer.SetFileName(dataPath)
    standard_writer.Execute(newSitkImage)



if __name__ == '__main__':

    path = os.path.abspath("/")
    inputName = '/'
    outputName = '/.nii.gz' #The forward slash is necessary to path+*Name work!

    sitkImage = sitk.ReadImage(path+inputName)
   
    # We use the sigmoid mask specifically for segmentation on porous media MRI
    # but feel free to import the functions and use other masks
    sigmoidMask = createSigmoidMask(sitk.Cast(sitkImage, sitk.sitkInt16))

    antsMask = convertSitkToAnts(sigmoidMask)
    antsImage = convertSitkToAnts(sitkImage)

    segmentationArray = getSegmentationArray(antsImage, antsMask)

    writeArrayImage(path+outputName, sitkImage, segmentationArray)

    
