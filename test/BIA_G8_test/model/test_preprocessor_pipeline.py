import os
import tempfile

from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor


def test():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tmp.toml")
        pp = (
            PreprocessorPipeline().
            add_step(
                get_preprocessor("normalize")().set_params()
            ).
            add_step(
                get_preprocessor("denoise (mean)")().set_params(footprint_length_width=5)
            )
        )
        pp.save(save_path)
        pp2 = PreprocessorPipeline.load(save_path)
        assert pp == pp2
        os.remove(save_path)
