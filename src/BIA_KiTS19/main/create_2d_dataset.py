import os
import shutil

import click
import tqdm

from BIA_KiTS19 import get_lh
from BIA_KiTS19.helper import dataset_helper, joblib_helper
from BIA_KiTS19.helper import io_helper, ndarray_helper

_lh = get_lh(__name__)


def slice_image_set(
        data_dir: str,
        image_set: dataset_helper.ImageSet
):
    for _axis in range(3):
        _axis_dir = os.path.join(data_dir, str(_axis), image_set.case_name)
        os.mkdir(_axis_dir)
        for i, (
                image_slice_np,
                mask_slice_np,
                image_slice_tf,
                mask_slice_tf
        ) in enumerate(zip(
            ndarray_helper.sample_along_np(image_set.np_image_final, axis=_axis),
            ndarray_helper.sample_along_np(image_set.np_mask_final, axis=_axis),
            ndarray_helper.sample_along_tensor(image_set.tensor_image_final, axis=_axis),
            ndarray_helper.sample_along_tensor(image_set.tensor_mask_final, axis=_axis),
        )):
            _lh.debug("Slicing %dth slice of case %s", i, image_set.case_name)
            io_helper.save_np_xz(image_slice_np, os.path.join(_axis_dir, f"{i}_image.npy.xz"))
            io_helper.save_np_xz(mask_slice_np, os.path.join(_axis_dir, f"{i}_mask.npy.xz"))
            io_helper.save_tensor_xz(image_slice_tf, os.path.join(_axis_dir, f"{i}_image.pt.xz"))
            io_helper.save_tensor_xz(mask_slice_tf, os.path.join(_axis_dir, f"{i}_mask.pt.xz"))
    image_set.clear_all_cache()


@click.command()
@click.option(
    "--data_dir",
    default=os.getcwd(),
    help="The directory where all cases (i.e., folders started with case_00000 -> case_00209) are stored"
)
def main(data_dir: str):
    _lh.info("Loading dataset at %s", data_dir)
    dataset = dataset_helper.DataSet(data_dir=data_dir)
    data_dir = os.path.join(data_dir, "2d")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)

    for axis in range(3):
        axis_dir = os.path.join(data_dir, str(axis))
        os.makedirs(axis_dir, exist_ok=True)
    joblib_helper.parallel_map(
        lambda image_set: slice_image_set(data_dir=data_dir, image_set=image_set),
        tqdm.tqdm(iterable=dataset, desc="Slicing images"),
        n_jobs=10
    )

    _lh.info("Loading dataset at %s FIN", data_dir)


if __name__ == "__main__":
    main()
