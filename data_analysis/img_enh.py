import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skiio
from scipy.signal import convolve2d


def gaussian_pyramid(image, kernel, levels):
    """
    A function to create a Gaussian pyramid of a defined number of levels and from a chosen kernel.
    :param image: The image we want to use of dimension (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Gaussian pyramid, an integer
    :return: The Gaussian pyramid, a list of numpy arrays
    """
    gauss_l = image
    pyramid = [image]
    for l in range(levels):
        gauss_l = downsample(gauss_l, kernel)
        pyramid.append(gauss_l)
    return pyramid


def laplacian_pyramid(image, kernel, levels):
    """
    A function to create a Laplacian pyramid of a defined number of levels and from a chosen kernel.
    :param image: The image we want to use of dimension (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Laplacian pyramid, an integer
    :return: The Laplacian pyramid, a list of numpy arrays
    """

    gauss = gaussian_pyramid(image, kernel, levels)
    pyramid = []
    for l in range(len(gauss) - 2, -1, -1):
        gauss_l1 = upsample(gauss[l + 1])
        if gauss_l1.shape[0] > gauss[l].shape[0]:
            gauss_l1 = np.delete(gauss_l1, -1, axis=0)
        if gauss_l1.shape[1] > gauss[l].shape[1]:
            gauss_l1 = np.delete(gauss_l1, -1, axis=1)
        lap_l = gauss[l] - gauss_l1
        pyramid.append(lap_l)
    return pyramid


def collapse_pyramid(lap_pyramid, gauss_pyramid):
    """
    A function to collapse a Laplacian pyramid in order to recover the enhanced image
    :param lap_pyramid: A Laplacian pyramid, a list of grayscale images, the last one in highest resolution
    :param gauss_pyramid: A Gaussian pyramid, a list of grayscale images, the last one in lowest resolution
    :return: A grayscale image
    """

    image = lap_pyramid[0]
    gauss = upsample(gauss_pyramid[-1])
    if gauss.shape[0] > image.shape[0]:
        gauss = np.delete(gauss, -1, axis=0)
    if gauss.shape[1] > image.shape[1]:
        gauss = np.delete(gauss, -1, axis=1)
    image = image + 0.5 * gauss
    for l in range(1, len(lap_pyramid), 1):
        pyr_upsampled = upsample(image)
        if pyr_upsampled.shape[0] > lap_pyramid[l].shape[0]:
            pyr_upsampled = np.delete(pyr_upsampled, -1, axis=0)
        if pyr_upsampled.shape[1] > lap_pyramid[l].shape[1]:
            pyr_upsampled = np.delete(pyr_upsampled, -1, axis=1)
        image = lap_pyramid[l] + pyr_upsampled
    return image


def smooth_gaussian_kernel(a):
    """
     A 5*5 gaussian kernel to perform smooth filtering.
    :param a: the coefficient of the smooth filter. A float usually within [0.3, 0.6]
    :return: A smoothing Gaussian kernel, a numpy array of shape (5,5)
    """
    w = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    kernel = np.outer(w, w)
    return kernel


def convolve(image, kernel):
    """
    A fonction to perform a 2D convolution operation over an image using a chosen kernel.
    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The convolution kernel of dimention (k,k)
    :return: The convolved image of dimension (N,M)
    """
    im_out = convolve2d(image, kernel, mode='same', boundary='symm')
    return im_out


def upsample(image):
    """
    :param image: The grayscale image we want to use of dimension (N,M)
    :param factor: The upsampling factor, an integer
    :return: The upsampled image of dimension (N*2,M*2)
    """

    kernel = smooth_gaussian_kernel(0.4)

    img_upsampled = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    img_upsampled[::2, ::2] = image[:, :]
    img_upsampled = 4 * convolve(img_upsampled, kernel)
    return img_upsampled


def downsample(image, kernel):
    """
    A function to downsample an image.
    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The Gaussian blurring kernel of dimention (k,k)
    :return: The downsampled image of dimension (N/2,M/2)
    """
    blur_image = convolve(image, kernel)
    img_downsampled = blur_image[::2, ::2]
    return img_downsampled


def my_sharp(image, kernel, levels):
    """
    A function to build the Gaussian and Laplacian pyramids of an image
    :param image: A grayscale image, a numpy array of floats within [0, 1] of shape (N, M)
    :param kernel: The Gaussian kernel used to build pyramids
    :param levels: The desired levels in the pyramids
    """

    # Building the Gaussian and Laplacian pyramids
    gauss_pyr = gaussian_pyramid(image, kernel, levels)
    lap_pyr = laplacian_pyramid(image, kernel, levels)
    collapsed_image = collapse_pyramid(lap_pyr, gauss_pyr)
    return collapsed_image


if __name__ == "__main__":
    kernel = smooth_gaussian_kernel(0.4)
    levels = 10
    image = skiio.imread("D:\BIA-G8\src\ipynb\COVID-19_Radiography_Dataset\images\COVID\COVID-1.png")
    imge_sharp = my_sharp(image, kernel, levels)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image[100:150, 50:200])
    ax[1].imshow(imge_sharp[100:150, 50:200])
    plt.show()
