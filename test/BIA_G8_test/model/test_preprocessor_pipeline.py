import os

from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor


def test():
    pp = (
        PreprocessorPipeline().
        add_step(
            get_preprocessor("normalize")().set_params()
        ).
        add_step(
            get_preprocessor("denoise (mean)")().set_params(footprint_length_width=5)
        )
    )
    pp.save("1.toml")
    pp2 = PreprocessorPipeline.load("1.toml")
    assert pp == pp2
    os.remove("1.toml")
