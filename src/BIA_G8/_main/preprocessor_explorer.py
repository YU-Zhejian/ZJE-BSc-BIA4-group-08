import click
import skimage.io as skiio
from matplotlib import pyplot as plt

from BIA_G8.helper import io_helper
from BIA_G8.helper.ndarray_helper import scale_np_array
from BIA_G8.model.preprocesor_pipeline import PreprocessorPipeline
from BIA_G8.model.preprocessor import get_preprocessor_name_descriptions, get_preprocessor


@click.command
@click.option("--input_path", help="Path to sample image")
@click.option("--pp_output_path", help="Output path of Preprocessor Pipeline Config.")
def setup_pp(
        input_path: str,
        pp_output_path: str
) -> PreprocessorPipeline:
    if input_path.endswith("npy.xz"):
        orig_img = io_helper.read_np_xz(input_path)
    else:
        orig_img = skiio.imread(input_path)
    orig_img = scale_np_array(orig_img)

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
        try:
            transformed_img = preprocessor.execute(transformed_img)
        except Exception as e:
            print(f"Exception {e} captured!")
            continue
        plt.imshow(
            transformed_img,
            cmap="bone"
        )
        plt.colorbar()
        plt.show()
        is_accepted = input("Accept? Y/N >") == "Y"
        if is_accepted:
            pp = pp.add_step(preprocessor)
            pp.save(pp_output_path)
        plt.close()
    print(f"Your configuration was saved to {pp_output_path}")
    return pp


if __name__ == '__main__':
    setup_pp()
