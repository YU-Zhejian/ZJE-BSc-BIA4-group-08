import doctest
from typing import List

import numpy as np
import numpy.typing as npt
import prettytable


def print_confusion_matrix(
        matrix: npt.NDArray,
        labels: List[str]
) -> str:
    """
    Print confusion matrix using ``prettytable`` package.

    Example:

    >>> print(print_confusion_matrix(np.array([[1, 0, 0], [0, 5, 2], [0, 0, 9]]), labels=["a", "b", "c"]))
    +-------+---+---+---+
    | title | a | b | c |
    +-------+---+---+---+
    |   a   | 1 | 0 | 0 |
    |   b   | 0 | 5 | 2 |
    |   c   | 0 | 0 | 9 |
    +-------+---+---+---+

    :param matrix: The confusion matrix.
    :param labels: Row and Column names.
    :return: Confusion matrix in string.
    """
    table_length = matrix.shape[0]
    field_names = list(map(labels.__getitem__, range(table_length)))
    pt = prettytable.PrettyTable(("title", *field_names))
    for i in range(table_length):
        pt.add_row((labels[i], *matrix[i]))
    return str(pt)


if __name__ == '__main__':
    doctest.testmod()
