"""SeedBundle loader: parses and validates the per-locale YAML seed files
that drive the dataset generator.

The pipeline owns no hardcoded data. Names, addresses, identifier tables,
slot-templates — all of it lives in `seeds/<locale>/*.yaml` and reaches
runtime through this module. Adding pools or templates is a YAML edit; no
Python change required.

Validation is strict via pydantic: typos, missing required fields, or
cross-references to undefined slots fail at load time, not at run time.
The smoke-test imports this module so misconfigurations surface in CI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Slot syntax in templates: ``{PLACEHOLDER_NAME}``. Anchored so partial
# matches like {NAME_AND_OTHER} can't sneak in.
_SLOT_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")

# Canonical placeholder taxonomy that templates may reference. Mirrors the
# regex/OPF policy taxonomy declared in ``src/traciora_pii/policies``.
# Listed here without brackets; the slot syntax is bracket-free.
_KNOWN_SLOTS = frozenset({
    "PERSON_NAME", "STREET_ADDRESS", "EMAIL",
    "RU_INN", "RU_SNILS", "PASSPORTNUM", "RU_OGRN", "RU_KPP",
    "RU_BANK_ACCOUNT", "VIN", "RU_DRV_LICENSE", "RU_STS",
    "RU_OSAGO",
    "DE_STEUER_ID", "DE_PERSONALAUSWEIS", "DE_PLZ_CITY",
    "US_SSN",
    "PHONE",
    "LICENSE_PLATE", "CARD", "IBAN", "IP", "DATE", "URL", "SECRET",
})


# ======================================================================
# Pydantic models — one per YAML file
# ======================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- names.yaml -----------------------------------------------------


class FeminineInflection(_Strict):
    suffix: str
    becomes: str


class NamesConfig(_Strict):
    first_names_male: List[str] = Field(default_factory=list)
    first_names_female: List[str] = Field(default_factory=list)
    last_names: List[str] = Field(default_factory=list)
    patronymics_male: List[str] = Field(default_factory=list)
    patronymics_female: List[str] = Field(default_factory=list)
    feminine_inflections: List[FeminineInflection] = Field(default_factory=list)
    short_format: Optional[str] = None

    @model_validator(mode="after")
    def _check_nonempty(self) -> "NamesConfig":
        if not self.last_names:
            raise ValueError("names.yaml: last_names must be non-empty")
        if not (self.first_names_male or self.first_names_female):
            raise ValueError(
                "names.yaml: at least one of first_names_male / "
                "first_names_female must be non-empty"
            )
        return self


# ---- addresses.yaml -------------------------------------------------


class CityStreets(_Strict):
    city: str
    streets: List[str]

    @field_validator("streets")
    @classmethod
    def _streets_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("city.streets must be non-empty")
        return v


class AddressesConfig(_Strict):
    city_streets: List[CityStreets] = Field(default_factory=list)
    formats: List[str] = Field(default_factory=list)
    house_range: Tuple[int, int] = (1, 250)
    apt_range: Tuple[int, int] = (1, 250)
    korp_range: Tuple[int, int] = (1, 10)
    # DE/EN locales may use a (zip, city, street) shape instead. Optional.
    plz_city_streets: Optional[List[Dict]] = None

    @model_validator(mode="after")
    def _check_addresses(self) -> "AddressesConfig":
        if not self.formats:
            raise ValueError("addresses.yaml: formats must be non-empty")
        if not (self.city_streets or self.plz_city_streets):
            raise ValueError(
                "addresses.yaml: at least one of city_streets / "
                "plz_city_streets must be non-empty"
            )
        for lo, hi in (self.house_range, self.apt_range, self.korp_range):
            if lo > hi or lo < 1:
                raise ValueError(f"invalid range: [{lo}, {hi}]")
        return self


# ---- identifiers.yaml -----------------------------------------------


class IdentifiersConfig(_Strict):
    """Catch-all bucket for locale-specific identifier knobs.

    Different locales populate different fields:
      RU: plate_letters, plate_regions, osago_series, mobile_prefixes, ...
      DE: phone_mobile_prefixes, phone_landline_city_codes, ...
      EN: phone_area_first_digit, plate_format, ...

    Every field is optional and the locale's generator only reads the
    ones it needs. New keys added to the YAML are surfaced via the
    ``extra`` dict so this schema does not need bumping for every
    locale-specific knob.
    """
    model_config = ConfigDict(extra="allow")


# ---- templates.yaml -------------------------------------------------


class Template(_Strict):
    text: str
    weight: int = 1

    @model_validator(mode="after")
    def _check_slots(self) -> "Template":
        if self.weight < 1:
            raise ValueError("template weight must be >= 1")
        slots = _SLOT_RE.findall(self.text)
        for slot in slots:
            if slot not in _KNOWN_SLOTS:
                raise ValueError(
                    f"template references unknown slot {{{slot}}}: {self.text!r}. "
                    f"Known slots: {sorted(_KNOWN_SLOTS)}"
                )
        if not slots:
            raise ValueError(
                f"template has no PII slots, would generate negative example: {self.text!r}"
            )
        return self

    @property
    def slot_names(self) -> Tuple[str, ...]:
        """Names of all PII slots in this template, in declaration order."""
        return tuple(_SLOT_RE.findall(self.text))


class CategoryConfig(_Strict):
    weight: int = 1
    templates: List[Template]

    @model_validator(mode="after")
    def _check_category(self) -> "CategoryConfig":
        if self.weight < 1:
            raise ValueError("category weight must be >= 1")
        if not self.templates:
            raise ValueError("category templates must be non-empty")
        return self


class TemplatesConfig(_Strict):
    categories: Dict[str, CategoryConfig]

    @model_validator(mode="after")
    def _check_categories(self) -> "TemplatesConfig":
        if not self.categories:
            raise ValueError("templates.yaml: categories must be non-empty")
        return self

    def all_slots_used(self) -> set:
        used = set()
        for cat in self.categories.values():
            for tpl in cat.templates:
                used.update(tpl.slot_names)
        return used


# ---- negatives.yaml -------------------------------------------------


class NegativesConfig(_Strict):
    """Hard-negative examples: PII-shaped patterns that must NOT trigger detection."""
    probability: float = 0.0
    categories: Dict[str, List[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_negatives(self) -> "NegativesConfig":
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"negatives.probability out of [0,1]: {self.probability}")
        for name, items in self.categories.items():
            if not items:
                raise ValueError(f"negatives category {name!r} is empty")
        return self

    def all_examples(self) -> List[str]:
        out: List[str] = []
        for items in self.categories.values():
            out.extend(items)
        return out


# ---- fillers.yaml ---------------------------------------------------


class FillersConfig(_Strict):
    """Per-category neutral filler sentences used to pad rendered templates."""
    prepend_probability: float = 0.0
    append_probability: float = 0.0
    categories: Dict[str, List[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_fillers(self) -> "FillersConfig":
        for name, p in (("prepend_probability", self.prepend_probability),
                        ("append_probability", self.append_probability)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"fillers.{name} out of [0,1]: {p}")
        for name, items in self.categories.items():
            if not items:
                raise ValueError(f"fillers category {name!r} is empty")
        return self

    def for_category(self, category: str) -> List[str]:
        """Fillers matching the given template category. Empty list if none."""
        return self.categories.get(category, [])


# ---- shared/secrets.yaml --------------------------------------------


class SecretShape(_Strict):
    name: str
    pattern: str
    body_charset: Optional[str] = None
    body_length: Optional[int] = None
    parts: Optional[Dict[str, Dict]] = None


class SecretsConfig(_Strict):
    shapes: List[SecretShape]


# ---- shared/domains.yaml --------------------------------------------


class DomainsConfig(_Strict):
    reserved: List[str]
    reserved_extra: List[str] = Field(default_factory=list)


# ---- shared/ip_blocks.yaml ------------------------------------------


class TestNet(_Strict):
    prefix: str
    name: str


class IpBlocksConfig(_Strict):
    test_nets: List[TestNet]
    host_octet_range: Tuple[int, int]


# ---- shared/ocr_confusables.yaml ------------------------------------


class ConfusablePair(_Strict):
    a: str
    b: str


class OcrConfusablesConfig(_Strict):
    multi_char: List[ConfusablePair] = Field(default_factory=list)
    single_char: List[ConfusablePair] = Field(default_factory=list)


# ======================================================================
# Top-level bundles
# ======================================================================


@dataclass(frozen=True)
class SharedSeeds:
    """Locale-neutral seed data shared across RU/DE/EN."""
    secrets: SecretsConfig
    domains: DomainsConfig
    ip_blocks: IpBlocksConfig
    ocr_confusables: OcrConfusablesConfig


@dataclass(frozen=True)
class SeedBundle:
    """All seed data for one locale, validated and ready to use."""
    locale: str
    names: NamesConfig
    addresses: AddressesConfig
    identifiers: IdentifiersConfig
    templates: TemplatesConfig
    negatives: NegativesConfig
    fillers: FillersConfig
    shared: SharedSeeds


# ======================================================================
# Loaders
# ======================================================================


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"seed file missing: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _seeds_root() -> Path:
    return Path(__file__).resolve().parent / "seeds"


def load_shared_seeds(seeds_dir: Optional[Path] = None) -> SharedSeeds:
    """Load locale-neutral seeds from ``seeds/shared/``."""
    base = (seeds_dir or _seeds_root()) / "shared"
    return SharedSeeds(
        secrets=SecretsConfig.model_validate(_read_yaml(base / "secrets.yaml")),
        domains=DomainsConfig.model_validate(_read_yaml(base / "domains.yaml")),
        ip_blocks=IpBlocksConfig.model_validate(_read_yaml(base / "ip_blocks.yaml")),
        ocr_confusables=OcrConfusablesConfig.model_validate(
            _read_yaml(base / "ocr_confusables.yaml"),
        ),
    )


def load_seed_bundle(locale: str, seeds_dir: Optional[Path] = None) -> SeedBundle:
    """Load and validate every YAML file for one locale.

    On any failure (missing file, schema violation, unknown slot referenced
    by a template) raises ``ValueError`` with a path-qualified message so
    the offending file is obvious.
    """
    base = (seeds_dir or _seeds_root()) / locale
    if not base.is_dir():
        raise FileNotFoundError(f"no seeds for locale {locale!r}: {base} not found")

    try:
        names = NamesConfig.model_validate(_read_yaml(base / "names.yaml"))
        addresses = AddressesConfig.model_validate(_read_yaml(base / "addresses.yaml"))
        identifiers = IdentifiersConfig.model_validate(_read_yaml(base / "identifiers.yaml"))
        templates = TemplatesConfig.model_validate(_read_yaml(base / "templates.yaml"))
        # Negatives and fillers are optional — absent file means an empty
        # config (probability=0, no examples). This keeps adding a new
        # locale frictionless: ship without negatives/fillers initially,
        # add them later as needed.
        negatives_path = base / "negatives.yaml"
        negatives = (
            NegativesConfig.model_validate(_read_yaml(negatives_path))
            if negatives_path.exists() else NegativesConfig()
        )
        fillers_path = base / "fillers.yaml"
        fillers = (
            FillersConfig.model_validate(_read_yaml(fillers_path))
            if fillers_path.exists() else FillersConfig()
        )
    except Exception as e:
        raise ValueError(f"failed to load seeds for locale {locale!r}: {e}") from e

    shared = load_shared_seeds(seeds_dir)
    return SeedBundle(
        locale=locale,
        names=names,
        addresses=addresses,
        identifiers=identifiers,
        templates=templates,
        negatives=negatives,
        fillers=fillers,
        shared=shared,
    )


def load_all_locales(seeds_dir: Optional[Path] = None) -> Dict[str, SeedBundle]:
    """Discover every locale subdir under ``seeds/`` and load its bundle.

    ``shared/`` is excluded.
    """
    base = seeds_dir or _seeds_root()
    out: Dict[str, SeedBundle] = {}
    for path in sorted(base.iterdir()):
        if not path.is_dir() or path.name == "shared":
            continue
        out[path.name] = load_seed_bundle(path.name, seeds_dir)
    return out
