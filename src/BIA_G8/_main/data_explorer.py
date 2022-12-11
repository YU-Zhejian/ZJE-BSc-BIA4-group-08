import click
import seaborn as sns
import skimage.transform as skitrans
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from BIA_G8.data_analysis.covid_dataset import CovidDataSet


@click.command()
@click.option("--dataset_path", help="Path to dataset")
def main(dataset_path: str):
    ds = CovidDataSet.parallel_from_directory(dataset_path).parallel_apply(
        lambda img: img[:, :, 0] if len(img.shape) == 3 else img
    ).parallel_apply(
        lambda img: skitrans.resize(img, (256, 256))
    )
    ds_tsne_transformed = TSNE(learning_rate=200, n_iter=1000, init="random").fit_transform(ds.sklearn_dataset[0])
    sns.scatterplot(
        x=ds_tsne_transformed[:, 0],
        y=ds_tsne_transformed[:, 1],
        hue=list(map(ds.decode, ds.sklearn_dataset[1]))
    )
    plt.show()


if __name__ == '__main__':
    main()
