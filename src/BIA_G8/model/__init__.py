"""
Model -- Models for Preprocessing and Classification

Here contains data structures and utilities for preprocessing and classification that should be called by the front end.
"""

from __future__ import annotations

from typing import TypeVar, Callable, Union, Generic

import BIA_G8

_lh = BIA_G8.get_lh(__name__)


class Unset:
    """
    Indicator indicating that an argument was not set.
    If instance of this class was passed,
    the default value of an argument should be used.

    :py:class:`Unset` should NOT be mixed with :py:class:`None`.
    They are created for different purposes!
    """

    def __repr__(self) -> str:
        return "unset"

    def __str__(self) -> str:
        return repr(self)


unset = Unset()
"""The only instance of :py:class:`Unset`."""


class LackingOptionalRequirementError(ValueError):
    """
    Exception indicating unmet dependencies.
    """

    def __init__(
            self,
            name: str,
            conda_channel: str,
            conda_name: str,
            pypi_name: str,
            url: str
    ):
        """
        :param name: Name of the dependency. For example, ``XGBoost``.
        :param conda_channel: Conda channel where this dependency can be found.
            For example, ``conda-forge``.
        :param conda_name: Name used to install using Conda. For example, ``py-xgboost-gpu``.
        :param pypi_name: Name used to install using ``pip``. For example, ``xgboost``.
        :param url: The project URL.
        """
        super().__init__(
            f"Dependency {name} not found!\n"
            f"Install it using: `conda install -c {conda_channel} {conda_name}`\n"
            f"Install it using: `pip install {pypi_name}`\n"
            f"See project URL: {url}"
        )


_ArgType = TypeVar("_ArgType")


def argument_string_to_string(instr: str) -> Union[Unset, str]:
    """
    Parse a string to integer. Return :py:class:`Unset` if string is empty.

    >>> argument_string_to_string("1")
    '1'
    >>> argument_string_to_string("")
    unset
    """
    return unset if instr == "" else instr


def argument_string_to_int(instr: str) -> Union[Unset, int]:
    """
    Parse a string to integer. Return :py:class:`Unset` if string is empty.

    >>> argument_string_to_int("1")
    1
    >>> argument_string_to_int("")
    unset
    """
    return unset if instr == "" else int(instr)


def argument_string_to_float(instr: str) -> Union[Unset, float]:
    """
    Parse a string to float. Return :py:class:`Unset` if string is empty.

    >>> argument_string_to_float("1")
    1.0
    >>> argument_string_to_float("")
    unset
    """
    return unset if instr == "" else float(instr)


class Argument(Generic[_ArgType]):
    """
    Parser and indicator of an argument. Used in frontend only.

    At the backend we declare an argument like this:

    >>> arg = Argument(name="arg", parse_str=argument_string_to_float, is_required=True, description="empty")
    >>> print(repr(arg))
    Argument ``arg`` (required: True) -- empty

    And at frontend it can be inspected like this:

    >>> arg.name
    'arg'

    And it can parse argument like this (by impleting the :py:func`object.__call__()` method.):

    >>> arg("3.1415")
    3.1415
    """
    _name: str
    _parse_str: Callable[[str], Union[_ArgType, Unset]]
    _is_required: bool
    _description: str

    def __init__(
            self,
            *,
            name: str,
            parse_str: Callable[[str], Union[_ArgType, Unset]],
            is_required: bool = False,
            description: str = "no argument description"
    ):
        self._name = name
        self._is_required = is_required
        self._parse_str = parse_str
        self._description = description

    @property
    def name(self) -> str:
        """Argument name"""
        return self._name

    @property
    def description(self) -> str:
        """Argument description"""
        return self._description

    @property
    def is_required(self) -> bool:
        """Whether the argument is required"""
        return self._is_required

    def __call__(self, in_str: str) -> Union[_ArgType, Unset]:
        return self._parse_str(in_str)

    def __repr__(self) -> str:
        return f"Argument ``{self._name}`` (required: {self._is_required}) -- {self._description}"

    def __str__(self) -> str:
        return repr(self)


class LackRequiredArgumentError(ValueError):
    """Argument Parser Exception for lack of required argument"""

    def __init__(self, argument: Argument):
        super().__init__(
            f"Lack required arguments: {argument}"
        )
