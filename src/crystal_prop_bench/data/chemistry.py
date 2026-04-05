"""Chemistry-family classification by dominant anion.

Standalone function — importable by evaluation, plotting, and analysis
code without adapter dependency.
"""

from pymatgen.core import Composition, Element

ANION_FAMILIES: dict[str, frozenset[Element]] = {
    "oxide": frozenset({Element("O")}),
    "sulfide": frozenset({Element("S")}),
    "nitride": frozenset({Element("N")}),
    "halide": frozenset({Element("F"), Element("Cl"), Element("Br"), Element("I")}),
}

# Reverse lookup: element -> family name
_ELEMENT_TO_FAMILY: dict[Element, str] = {}
for family_name, elements in ANION_FAMILIES.items():
    for el in elements:
        _ELEMENT_TO_FAMILY[el] = family_name


def classify_chemistry_family(
    composition: Composition,
    purity_threshold: float = 0.80,
) -> str | None:
    """Classify crystal by dominant anion.

    Returns 'oxide', 'sulfide', 'nitride', 'halide', or None
    if no anion family exceeds the purity threshold.

    Parameters
    ----------
    composition : Composition
        Pymatgen Composition object.
    purity_threshold : float
        Minimum fraction of anion sites that must belong to one family.
        Default 0.80 per DECISIONS.md.

    Returns
    -------
    str or None
        Family name, or None if below threshold or no recognized anions.
    """
    el_amounts = composition.get_el_amt_dict()

    # Accumulate anion amounts per family
    family_amounts: dict[str, float] = {}
    total_anion_amount = 0.0

    for el, amt in el_amounts.items():
        el_obj = Element(el) if isinstance(el, str) else el
        if el_obj in _ELEMENT_TO_FAMILY:
            family = _ELEMENT_TO_FAMILY[el_obj]
            family_amounts[family] = family_amounts.get(family, 0.0) + amt
            total_anion_amount += amt

    if total_anion_amount == 0.0:
        return None

    # Find dominant family
    for family, amt in sorted(
        family_amounts.items(), key=lambda x: x[1], reverse=True
    ):
        fraction = amt / total_anion_amount
        if fraction >= purity_threshold:
            return family

    return None
