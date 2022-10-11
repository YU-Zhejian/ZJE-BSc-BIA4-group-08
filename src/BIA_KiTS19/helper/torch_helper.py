import torch.utils.data as tud
from torch.utils.data.dataset import T_co

from BIA_KiTS19.helper import dataset_helper


class KiTS19DataSet(tud.Dataset):
    def __getitem__(self, index) -> T_co:
        ...  # TODO

    _dataset: dataset_helper.DataSet

    def __init__(self, dataset: dataset_helper.DataSet):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)
