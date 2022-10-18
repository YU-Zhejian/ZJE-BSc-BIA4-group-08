from typing import TypeVar, Callable, Any

from BIA_G8 import get_lh

_CacheType = TypeVar("_CacheType")

_lh = get_lh(__name__)


class AbstractCachedLazyEvaluatedLinkedChain:
    """
    A lazy-evaluated dataset with disk cache, with following perspectives:

    - The dataset have several properties.
    - Properties can be generated from disk cache, or from some other properties, or both.
    """

    def _ensure_from_cache_file(
            self,
            cache_property_name: str,
            cache_filename: str,
            cache_file_reader: Callable[[str], _CacheType]
    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache.

        :param cache_property_name: Name of property cache.
        :param cache_filename: Name of on-disk cache.
        :param cache_file_reader: Function that loads the file.
        :return: The desired property, initialized.
        """
        _lh.debug("Request property %s", cache_property_name)
        if self.__getattribute__(cache_property_name) is None:
            _lh.debug("Request property %s -- Load from %s", cache_property_name, cache_filename)
            if os.path.exists(cache_filename):
                self.__setattr__(cache_property_name, cache_file_reader(cache_filename))
                _lh.debug("Request property %s -- Load from %s FIN", cache_property_name, cache_filename)
            else:
                _lh.debug("Request property %s -- Load from %s ERR", cache_property_name, cache_filename)
                raise FileNotFoundError(f"Read desired property {cache_property_name} from {cache_filename} failed")
        _lh.debug("Request property %s FIN", cache_property_name)
        return self.__getattribute__(cache_property_name)

    def _ensure_from_previous_step(
            self,
            cache_property_name: str,
            prev_step_property_name: str,
            transform_from_previous_step: Callable[[Any], _CacheType],

    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache or from a previous step.

        :param cache_property_name: Name of property cache.
        :param prev_step_property_name: Property of previous step.
        :param transform_from_previous_step: Transformer function that transforms data from previous step to this step.
        :return: The desired property, initialized.
        """
        _lh.debug("Request property %s", cache_property_name)
        if self.__getattribute__(cache_property_name) is None:
            _lh.debug("Request property %s -- generate from %s", cache_property_name, prev_step_property_name)
            self.__setattr__(
                cache_property_name,
                transform_from_previous_step(self.__getattribute__(prev_step_property_name))
            )
            _lh.debug("Request property %s -- generate from %s FIN", cache_property_name, prev_step_property_name)
        _lh.debug("Request property %s FIN", cache_property_name)
        return self.__getattribute__(cache_property_name)

    def _ensure_from_cache_and_previous_step(
            self,
            cache_property_name: str,
            cache_filename: str,
            cache_file_reader: Callable[[str], _CacheType],
            cache_file_writer: Callable[[_CacheType, str], None],
            prev_step_property_name: str,
            transform_from_previous_step: Callable[[Any], _CacheType],

    ) -> _CacheType:
        """
        Ensure the existence of some property, which can be generated from disk cache or from a previous step.

        :param cache_property_name: Name of property cache.
        :param cache_filename: Name of on-disk cache.
        :param cache_file_reader: Function that loads the file.
        :param cache_file_writer: Function that writes data to on-disk cache.
        :param prev_step_property_name: Property of previous step.
        :param transform_from_previous_step: Transformer function that transforms data from previous step to this step.
        :return: The desired property, initialized.
        """
        try:
            _property: _CacheType = self._ensure_from_cache_file(
                cache_property_name,
                cache_filename,
                cache_file_reader
            )
        except FileNotFoundError:
            _property: _CacheType = self._ensure_from_previous_step(
                cache_property_name,
                prev_step_property_name,
                transform_from_previous_step
            )
            _lh.debug("Request property %s -- save to %s", cache_property_name, cache_filename)
            cache_file_writer(_property, cache_filename)
            _lh.debug("Request property %s -- save to %s FIN", cache_property_name, cache_filename)
        return _property

