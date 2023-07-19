#Author: Gustavo Solcia
#Email: gustavo.solcia@usp.br

"""Creates an nifti cylinder image for a test case.
"""

import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

def fill_image(image, x_center, y_center):
    """Fills an image with a cylinder.

    Parameters:
        image: np.array
            The image to be filled.
        x_center: int
            The x coordinate of the cylinder center.
        y_center: int
            The y coordinate of the cylinder center.
        z_center: int
            The z coordinate of the cylinder center.
        radius: float
            The radius of the cylinder.
        height: float
            The height of the cylinder.
        
    Returns:
        image: np.array
            The image filled with the cylinder.
    """

    # Get the image size
    x_size = image.shape[0]
    y_size = image.shape[1]
    z_size = image.shape[2]


    # Fil the image with cylinder based on % of the maximum radius and height in a 1/2 proportion and square for segmentation
    radius = x_size*0.4
    z_start = z_size*0.1
    z_end = z_size*0.9
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if (x >= x_center-radius and x <= x_center+radius) and (y >= y_center-radius and y <= y_center+radius) and (z >= z_start and z <= z_end):
                    image[x,y,z] = 1
                if (x-x_center)**2 + (y-y_center)**2 <= radius**2 and (z >= z_start and z <= z_end):
                    image[x,y,z] = 5


    return image

def add_cosine_field(image, plot_field):
    """Add cosine field in z direction to the image.

    Parameters
    ----------
    image : np.array
        The image to be noised.
    plot_field : bool
        If True, plots the field.

    Returns
    -------
    image_field : np.array
        The image with the cosine field.
    """

    # Get the image size
    x_size = image.shape[0]
    y_size = image.shape[1]
    z_size = image.shape[2]

    # Create the field
    field = np.zeros((x_size, y_size, z_size))
    for z in range(z_size):
        field[:,:,z] = np.cos(z/z_size*2*np.pi)

    # Add the field to the image
    image_field = image + field

    if plot_field:
        plt.figure(figsize=(10,10))
        sns.set_style('ticks')
        sns.set_context('talk')
        plt.imshow(field[:,y_size//2,:], cmap='gray')
        plt.colorbar()
        plt.show()

    return image_field

def add_rician_noise(image, noise_level, plot_noise):
    """Adds Rician noise to an image according to the specified noise level.
    
    Parameters
    ----------
    image : np.array
        The image to be noised.
    noise_level : float
        The noise level between 0 and 1.
    plot_noise : bool
        If True, plots the noise histogram.
    
    Returns
    -------
    image_noise : np.array
        The noised image.
    """

    # Get the image size
    x_size = image.shape[0]
    y_size = image.shape[1]
    z_size = image.shape[2]

    # Create the noise
    noise_real = np.zeros((x_size, y_size, z_size))
    noise_img = np.zeros((x_size, y_size, z_size))

    noise_real = np.random.normal(0, noise_level, (x_size, y_size, z_size))
    noise_img = np.random.normal(0, noise_level, (x_size, y_size, z_size))
    noise = np.sqrt(noise_real**2 + noise_img**2)

    # Add the noise to the image
    image_noise = image + noise

    if plot_noise:
        plt.figure(figsize=(10,10))
        sns.set_style('ticks')
        sns.set_context('talk')
        plt.hist(noise.flatten(), bins=50, color='k')
        plt.ylabel('Frequency')
        plt.xlabel('Noise Intensity')
        plt.xlim([0, 5])
        plt.show()

    return image_noise

if __name__ == '__main__':

    # Define the image size
    x_size = 126
    y_size = 126
    z_size = 256

    # Define the center of the cylinder
    x_center = x_size/2
    y_center = y_size/2

    # Create the image
    image = np.zeros((x_size, y_size, z_size))
    image = fill_image(image, x_center, y_center)

    image = add_cosine_field(image, plot_field=True)

    # Add noise to the image
    image = add_rician_noise(image, 1, plot_noise=True)

    # Save the image
    nib.save(nib.Nifti1Image(image, np.eye(4)), 'testSample.nii.gz')
