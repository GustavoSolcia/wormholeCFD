#Author: Gustavo Solcia
#E-mail: gustavo.solcia@usp.br

"""Import stl from boundary surface and save boundary points coordinates.
"""
import csv
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

def visualize_boundaryPoints(points):
    """
    Simple visualization from boundary points.

    Parameter
    ---------
    points: array
        Points from given boundary file in stl format.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], color='k')
    plt.show()


if __name__=='__main__':

    surfaceMesh = mesh.Mesh.from_file('stlRecon/IXI160-LICA.stl')
    shape = np.shape(surfaceMesh.vectors)
    points = np.unique(surfaceMesh.vectors.reshape([shape[0]*3,3]), axis=0)

    f = open('boundaryPoints_LICA','w', encoding='UTF8')
    f.write(str(len(points))+'\n')
    f.write('('+'\n')
    for point in points:
        f.write('('+str(point[0])+' '+str(point[1])+' '+str(point[2])+')'+'\n')
    f.write(')'+'\n')
    f.close()

    visualize_boundaryPoints(points)
