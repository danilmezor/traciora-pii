"""Category exclusions shared by regex and second-pass masking."""
from collections.abc import Iterable, Set
from functools import lru_cache
from .policies.schemas import VaultCategory


@lru_cache(maxsize=1)
def _placeholder_categories() -> dict[str, VaultCategory]:
    from .policies.loader import load_multi_regex_policy
    return load_multi_regex_policy(["en", "de", "ru"]).placeholder_to_category


def resolve_exclusions(
    categories: Iterable[str] | None = None,
) -> frozenset[VaultCategory]:
    if categories is None:
        from .config import get_config
        categories = get_config().detector.excluded_categories
    return frozenset(VaultCategory(category) for category in categories)


def category_is_excluded(
    placeholder: str, categories: Set[VaultCategory],
) -> bool:
    return bool(categories) and _placeholder_categories().get(placeholder) in categories
