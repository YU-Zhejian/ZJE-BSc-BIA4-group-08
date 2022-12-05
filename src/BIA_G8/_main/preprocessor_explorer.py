import numpy.typing as npt
import skimage
from matplotlib import pyplot as plt

from BIA_G8.helper.io_helper import read_np_xz
from BIA_G8.helper.ndarray_helper import describe, scale_np_array
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor_name_descriptions, get_preprocessor


def setup_pp(
        orig_img: npt.NDArray,
        pp_output_path: str
) -> PreprocessorPipeline:
    pp = PreprocessorPipeline()
    cached_preprocessor_name_description = {
        preprocessor_id: name_description for preprocessor_id, name_description in
        enumerate(get_preprocessor_name_descriptions())
    }

    while True:
        print("Preprocessor Ready. Available Preprocessors:")
        for preprocessor_id, (preprocessor_name, preprocessor_description) \
                in cached_preprocessor_name_description.items():
            print(f"\t{preprocessor_id}: {preprocessor_name} -- {preprocessor_description}")
        selected_preprocessor_id = int(input("Enter Preprocessor ID (-1 to exit) >"))
        if selected_preprocessor_id == -1:
            break
        preprocessor = get_preprocessor(cached_preprocessor_name_description[selected_preprocessor_id][0])()
        print(f"Selected preprocessor {preprocessor.name}.")
        input_kwds = {}
        for argument in preprocessor.arguments:
            print(f"{argument}")
            argument_value = input(f"Argument Value for {argument.name} (blank for default) >")
            input_kwds[argument.name] = argument_value
        try:
            preprocessor = preprocessor.set_params(**input_kwds)
        except Exception as e:
            print(f"Exception {e} captured!")
            continue
        print(f"Preprocessor {repr(preprocessor)}")
        orig_img_copy = orig_img.copy()
        transformed_img = pp.execute(orig_img_copy)
        transformed_img = preprocessor.execute(transformed_img)
        plt.imshow(transformed_img)
        plt.show()
        is_accepted = input("Accept? Y/N >") == "Y"
        if is_accepted:
            pp = pp.add_step(preprocessor)
            pp.save(pp_output_path)
        plt.close()
    print(f"Your configuration was saved to {pp_output_path}")
    return pp


if __name__ == '__main__':
    input_path: str = "/home/yuzj/Documents/2022-23-Group-08/data_analysis/covid_image_new/Viral Pneumonia/Viral Pneumonia-1029.npy.xz"
    image = scale_np_array(read_np_xz(input_path), (-1, 1))
    print(describe(image))
    setup_pp(
        skimage.img_as_uint(image),
        "../../BIA_G8_DATA_ANALYSIS/explore.toml"
    )
