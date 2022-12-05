from __future__ import annotations

from typing import TypeVar, Callable, Union

import BIA_G8

_lh = BIA_G8.get_lh(__name__)


class Unset:
    pass


unset = Unset()


class LackingOptionalRequirementError(ValueError):
    def __init__(
            self,
            name: str,
            conda_channel: str,
            conda_name: str,
            pypi_name: str,
            url: str
    ):
        super().__init__(
            f"Dependency {name} not found!\n"
            f"Install it using: `conda install -c {conda_channel} {conda_name}`\n"
            f"Install it using: `pip install {pypi_name}`\n"
            f"See project URL: {url}"
        )


_ArgType = TypeVar("_ArgType")


def argument_string_to_int(instr: str) -> Union[Unset, int]:
    return unset if instr == "" else int(instr)


def argument_string_to_float(instr: str) -> Union[Unset, float]:
    return unset if instr == "" else float(instr)


class Argument:
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
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_required(self) -> bool:
        return self._is_required

    def __call__(self, in_str: str) -> Union[_ArgType, Unset]:
        return self._parse_str(in_str)

    def __repr__(self) -> str:
        return f"Argument {self._name} (required: {self._is_required}) -- {self._description}"


class LackRequiredArgumentError(ValueError):
    """Argument Parser Exception for lack of required argument"""

    def __init__(self, argument: Argument):
        super().__init__(
            f"Lack required arguments: {argument}"
        )
