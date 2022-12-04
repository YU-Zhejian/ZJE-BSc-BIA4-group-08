import numpy.typing as npt
import skimage.io as skiio
from matplotlib import pyplot as plt

from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor_name_descriptions, get_preprocessor, LackRequiredArgumentError


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
        print(f"Selected preprocessor {preprocessor.name}. Available arguments: {list(preprocessor.argument_names)}")
        input_kwds = {}
        for argument_name in preprocessor.argument_names:
            argument_value = input(f"Argument Valye for {argument_name} (blank for default) >")
            input_kwds[argument_name] = argument_value
        try:
            preprocessor = preprocessor.set_params(**input_kwds)
        except LackRequiredArgumentError as e:
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
    input_path: str = "/home/yuzj/Documents/2022-23-Group-08/doc/ipynb/sample_covid_image/COVID-19/e1087c6c-582b-4384-9c6f-b84784512ddc.png"
    setup_pp(skiio.imread(input_path), "1.toml")
