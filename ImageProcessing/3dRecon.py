# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""3D reconstruction using a segmentation image. We apply Marching Cubes algorithm and a 
   surface smoothing from the largest connected region. LOOK AT YOUR DATA: for some cases 
   the surface smoothing can shrink parts of your object.
"""

import os
import vtk
import SimpleITK as sitk

def applyMarchingCubes(image, threshold, transformCoord, QFormMatrix):

    """Wrapper function to apply vtk marching cubes algorithm. If transformCoord ==True applies vtk transform filter for alignment between vtkImageData and vtkPolyData. More info: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html

    Parameters
    ----------
    image: vtkImage
        vtkImage from vtkNIFTIImageReader
    threshold: float
        threshold for binarization purposes

    Returns
    -------
    largestRegion: vtkPolyData
        vtk data object that represents a geometric structure with vertices, lines, polygons...

    """
    contourNumber = 0

    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(image)
    marchingCubes.ComputeNormalsOn()
    marchingCubes.ComputeGradientsOn()
    marchingCubes.SetValue(contourNumber, threshold)
    marchingCubes.Update()
    
    if transformCoord==True:
        for i in range(2):
            for j in [0,1,2,3]:
                QFormMatrix.SetElement(i,j,-QFormMatrix.GetElement(i,j))
        transform = vtk.vtkTransform()
        transform.SetMatrix(QFormMatrix)
        transform.Update()

        transformPoly = vtk.vtkTransformPolyDataFilter()
        transformPoly.SetInputConnection(marchingCubes.GetOutputPort())
        transformPoly.SetTransform(transform)
        transformPoly.Update()

        mcPoly = transformPoly.GetOutput()
    else:
        mcPoly = marchingCubes.GetOutput()

    largestRegion = getLargestRegion(mcPoly)

    return largestRegion


def getLargestRegion(poly):

    """Function to get largest connected region from vtk poly data.

    Parameters
    ----------
    poly: vtkPolyData
        vtk data object that represents a geometric structure with vertices, lines, polygons...

    Returns
    -------
    largestRegion: vtkPolyData
        largest connected region from poly input

    """
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(poly)
    connectivityFilter.SetExtractionModeToLargestRegion()
    connectivityFilter.Update()

    largestRegion = connectivityFilter.GetOutput()
    
    return largestRegion

def applyPolyFilter(poly):

    """Function that apply a surface vtk poly data filter. 

    Parameters
    ----------
    poly: vtkPolyData
        vtk data object that represents a geometric structure with vertices, lines, polygons...

    Returns
    -------
    smoothPoly: vtkPolyData
        smooth surface from poly input

    """

    #These are parameters that worked fine for most of my applications.
    #However, if you are having shrinking problems: 
    #-First, I would consider a higher passBand (e. g., 0.3, 0.4, 0.5, etc...).
    #-Second, with a different passBand, I would increase the numberOfIterations
    #and gradually decrease that number (but never going less than 100 iterations).
    numberOfIterations = 100
    passBand = 0.25
    featureAngle = 120.0
    
    polyFilter = vtk.vtkWindowedSincPolyDataFilter()
    polyFilter.SetInputData(poly)
    polyFilter.SetNumberOfIterations(numberOfIterations)
    polyFilter.SetPassBand(passBand)
    polyFilter.SetFeatureAngle(featureAngle)
    polyFilter.Update()

    smoothPoly = polyFilter.GetOutput()
    return smoothPoly

def readImage(path, name):

    """vtkNIFTI image reader wrapper function.

    Parameters
    ----------
    path: string
        String containing a path to the data directory
    name: string
        String containing the data or sample name

    Returns
    -------
    image: vtkNIFTIImage
        Desired image from path+name.
    QFormMatrix: vtkMatrix
        QFormMatrix for vtk transform filter.
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(path+name)
    reader.Update()

    image = reader.GetOutput()

    QFormMatrix = reader.GetQFormMatrix()

    return image, QFormMatrix

def writeSTL(path, name, poly):

    """STL writting function for vtkPolyData.

    Parameters
    ----------
    path: string
        String containing a path to the data directory
    name: string
        String containing the data or sample name
    poly: vtkPolyData
        vtk data object that represents a geometric structure with vertices, lines, polygons...

    """

    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(path+name)
    writer.Update()

if __name__=='__main__':
    
    sample = ''
    inputPath = os.path.abspath(''+sample+'')
    outputPath = os.path.abspath(''+sample+'/')
    inputName = '/Atropos_'+sample+'.nii.gz'
    cubesOutputName = '/cubes_'+sample+'.stl'
    smoothOutputName = '/smooth_'+sample+'.stl'

    threshold = 2.5
    transformCoord=True

    vtkImage, QFormMatrix = readImage(inputPath, inputName)

    mcPoly = applyMarchingCubes(vtkImage, threshold, transformCoord, QFormMatrix)

    polyFiltered = applyPolyFilter(mcPoly)

    writeSTL(outputPath,cubesOutputName, mcPoly)

    writeSTL(outputPath,smoothOutputName, polyFiltered)
