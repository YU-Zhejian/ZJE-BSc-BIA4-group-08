import operator
from functools import reduce

import numpy as np
import skimage

stride = skimage.img_as_int(skimage.img_as_float(
    np.array(reduce(operator.add, map(lambda x: [[x] * 100] * 10, range(0, 100, 10))))
))
