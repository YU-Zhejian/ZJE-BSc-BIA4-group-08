from typing import Dict


class AnalysisConfiguration:
    _database_path:str
    _encoder_dict: Dict[str, int]

    @property
    def database_path(self) -> str:
        return self._database_path

    @property
    def encoder_dict(self) -> Dict[str, int]:
        return dict(self._encoder_dict)


