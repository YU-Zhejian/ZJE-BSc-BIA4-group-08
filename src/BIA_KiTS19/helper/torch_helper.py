from typing import Dict

import torch.utils.data as tud

from BIA_KiTS19.helper import dataset_helper


class KiTS19DataSet(tud.Dataset):
    def __getitem__(self, index: int) -> dataset_helper.ImageSet:
        return self._dataset[index]

    _dataset: Dict[int, dataset_helper.ImageSet]

    def __init__(self, dataset: dataset_helper.DataSet):
        self._dataset = {i: v for i, v in enumerate(dataset)}

    def __len__(self) -> int:
        return len(self._dataset)
