from __future__ import annotations

import BIA_G8

_lh = BIA_G8.get_lh(__name__)


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
