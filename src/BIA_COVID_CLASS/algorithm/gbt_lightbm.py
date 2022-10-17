import multiprocessing

import ray
from ray.air.config import ScalingConfig
from ray.train.lightgbm import LightGBMTrainer

from BIA_COVID_CLASS.covid_helper import covid_dataset
from BIA_G8 import get_lh

_lh = get_lh(__name__)

if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init()
    orig_num_blocks = multiprocessing.cpu_count()
    df = covid_dataset.get_ray_dataset(
        "/media/yuzj/BUP/covid19-database-np",
        size=9
    ).repartition(
        multiprocessing.cpu_count()
    )
    orig_len = df.count()
    train_dataset, valid_dataset = df.train_test_split(test_size=0.3)
    # Following step is to make parquet not empty, otherwise might raise errors.
    train_dataset = train_dataset.repartition(
        num_blocks=int(orig_num_blocks * train_dataset.count() / orig_len + 1),
        shuffle=True
    )
    valid_dataset = valid_dataset.repartition(
        num_blocks=int(orig_num_blocks * valid_dataset.count() / orig_len + 1),
        shuffle=True
    )
    sc = ScalingConfig(
        num_workers=(multiprocessing.cpu_count() - 1) // 2,
        use_gpu=False,
    )

    lightbm_trainer = LightGBMTrainer(
        scaling_config=sc,
        label_column="label",
        num_boost_round=200,
        params={
            "objective": "multiclass",
            "metric": ["multi_error"],
            "num_class": 3
        },
        datasets={
            "train": valid_dataset,
            "valid": train_dataset
        }
    )
    lightbm_result = lightbm_trainer.fit()
    print(lightbm_result.metrics_dataframe.tail(n=10))
