# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import skimage.filters as skifil
import skimage.io as skiio
import skimage.transform as skitrans
from skimage import restoration
from skimage.filters.rank import mean as skimean

# TODO: the user should pass the image directory
input_filename = "str"
img = skiio.imread(input_filename)
if len(img.shape) == 2:
    # the image size is rescaled to (256,256) to suit our software
    img = skitrans.resize(img, (256, 256))
    plt.imshow(img, cmap="gray")
# TODO: print ("The image is sucessfully imported and click here to preprocess it or click here to analyze it directly")
if len(img.shape) == 3:
    # TODO: print("Our software support gray scale image only, if it is a gray scale image click convert, if it is not please select another image")
    img = img[:, :, 0]
    img = skitrans.resize(img, (256, 256))
    plt.imshow(img, cmap="gray")
else:
    print("This is not a supported format, please select another image")

# TODO: quality control to ensure it is a front chest X-ray image

# Preprocess the image
# Enhance the image and prevent under exposure and over exposure
q2, q98 = np.percentile(img, (2, 98))
img = skiexp.rescale_intensity(img, in_range=(q2, q98))
img = skiexp.equalize_adapthist(img)

# Denoise the image: the user should choose which the filter to remove noise.
# Suggestions: for salt-pepper noise, median filter works well. for gaussian noise, gaussian filter workers better
# the user should type the size of the footprint as (int,int) for median and mean filter
footprint = np.ones((int, int))
# median filter
img = skifil.median(img, footprint=footprint)
# mean filter
img = skimean(img, footprint=footprint)
# gaussain filter the user should import sigma
sigma = int
img = skifil.gaussian(img, sigma=sigma)

# sharpen the image: the user should import the radius and amount of this unsharp filter
amount = float > 0
radius = float > 0
img = skifil.unsharp_mask(img, radius=radius, amount=amount)

# deblur the image: the user can deblur the image by setting the kernalsize and
kernalsize = (int, int)
balance = float
kernal = np.ones(kernalsize)
img = restoration.wiener(img, kernal / np.size(kernal), balance)

img_enh = img

# analyze with img_enh
