from typing import List

import numpy.typing as npt
import prettytable


def print_confusion_matrix(
        matrix: npt.NDArray,
        labels: List[str]
) -> str:
    table_length = matrix.shape[0]
    field_names = list(map(labels.__getitem__, range(table_length)))
    pt = prettytable.PrettyTable(("title", *field_names))
    for i in range(table_length):
        pt.add_row((labels[i], *matrix[i]))
    return str(pt)
