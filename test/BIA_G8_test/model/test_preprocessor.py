import numpy as np

from BIA_G8.model.preprocessor import get_preprocessor


def test():
    assert get_preprocessor("dimension reduction")().execute(np.zeros(shape=(1024, 1024))).shape == (256, 256)
    dnm = get_preprocessor("denoise (mean)")()
    assert list(dnm.argument_names) == ['footprint_length_width']
    dnm = dnm.set_params(footprint_length_width=5, aaa=6)
    assert dnm._parsed_kwargs == {'footprint_length_width': 5}
    assert dnm.execute(np.zeros(shape=(256, 256))).shape == (256, 256)
