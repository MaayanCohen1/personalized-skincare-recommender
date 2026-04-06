"""Test suite for matching_service.rules_engine — fit-scoring pipeline.

Covers: safety filtering, deterministic fit scoring, category-aware
selection, active-conflict avoidance, and the full match_products pipeline.
"""

import json

import pytest

from shared.models import Product, ProductCategory, UserConstraints
from matching_service.rules_engine import (
    _expand_sensitivities,
    _pick_with_active_policy,
    active_families,
    build_routine,
    build_routine_rationale,
    filter_safe_products,
    match_products,
    score_product_fit,
    select_balanced_routine,
    select_best_per_category,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def standard_constraints() -> UserConstraints:
    return UserConstraints(
        request_id="req-standard", sensitivities=[], max_products=5,
    )


@pytest.fixture
def fragrance_sensitive_constraints() -> UserConstraints:
    return UserConstraints(
        request_id="req-fragrance", sensitivities=["fragrance"], max_products=5,
    )


@pytest.fixture
def multi_sensitivity_constraints() -> UserConstraints:
    return UserConstraints(
        request_id="req-multi", sensitivities=["fragrance", "alcohol"], max_products=5,
    )


@pytest.fixture
def tight_max_constraints() -> UserConstraints:
    return UserConstraints(
        request_id="req-tight", sensitivities=[], max_products=3,
    )


@pytest.fixture
def essential_catalog() -> list[Product]:
    return [
        Product(id="cleanser-basic", name="Basic Cleanser",
                category=ProductCategory.CLEANSER,
                ingredients=["water", "glycerin"], description="Gentle daily cleanser."),
        Product(id="moisturizer-basic", name="Basic Moisturizer",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "shea butter"], description="Hydrating daily moisturizer."),
        Product(id="spf-basic", name="Basic SPF",
                category=ProductCategory.SPF,
                ingredients=["zinc oxide", "water"], description="Broad-spectrum SPF 50."),
    ]


@pytest.fixture
def mixed_catalog(essential_catalog: list[Product]) -> list[Product]:
    optional = [
        Product(id="serum-vitamin-c", name="Vitamin C Serum",
                category=ProductCategory.SERUM,
                ingredients=["ascorbic acid", "water"],
                description="Brightening vitamin C serum."),
        Product(id="toner-gentle", name="Gentle Toner",
                category=ProductCategory.TONER,
                ingredients=["witch hazel", "water"],
                description="Alcohol-free balancing toner."),
        Product(id="face-oil-other", name="Rosehip Face Oil",
                category=ProductCategory.OTHER,
                ingredients=["rosehip oil"],
                description="Nourishing face oil."),
    ]
    return essential_catalog + optional


@pytest.fixture
def scented_catalog() -> list[Product]:
    return [
        Product(id="scented-moisturizer", name="Scented Moisturizer",
                category=ProductCategory.MOISTURIZER,
                ingredients=["fragrance", "water", "glycerin"],
                description="Floral-scented moisturizer."),
        Product(id="unscented-cleanser", name="Unscented Cleanser",
                category=ProductCategory.CLEANSER,
                ingredients=["water", "glycerin"],
                description="Fragrance-free cleanser."),
    ]


# ---------------------------------------------------------------------------
# filter_safe_products — allergy filtering
# ---------------------------------------------------------------------------


def test_product_with_sensitizing_ingredient_is_removed(
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    products = [
        Product(id="bad-cream", name="Bad Cream",
                category=ProductCategory.MOISTURIZER,
                ingredients=["fragrance", "water"], description=""),
    ]
    result = filter_safe_products(products, fragrance_sensitive_constraints)
    assert result == []


def test_product_without_sensitizing_ingredients_passes(
    scented_catalog: list[Product],
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    result = filter_safe_products(scented_catalog, fragrance_sensitive_constraints)
    assert len(result) == 1
    assert result[0].id == "unscented-cleanser"


def test_filtering_is_case_insensitive() -> None:
    constraints = UserConstraints(
        request_id="req-case", sensitivities=["Fragrance"], max_products=5,
    )
    products = [
        Product(id="scented", name="Scented Serum",
                category=ProductCategory.SERUM,
                ingredients=["Fragrance", "niacinamide"], description=""),
    ]
    result = filter_safe_products(products, constraints)
    assert result == []


def test_multiple_products_only_unsafe_removed(
    multi_sensitivity_constraints: UserConstraints,
) -> None:
    products = [
        Product(id="p1", name="P1", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="p2", name="P2", category=ProductCategory.SERUM,
                ingredients=["alcohol", "fragrance"], description=""),
        Product(id="p3", name="P3", category=ProductCategory.MOISTURIZER,
                ingredients=["glycerin"], description=""),
        Product(id="p4", name="P4", category=ProductCategory.TONER,
                ingredients=["alcohol"], description=""),
        Product(id="p5", name="P5", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = filter_safe_products(products, multi_sensitivity_constraints)
    assert len(result) == 3
    assert {p.id for p in result} == {"p1", "p3", "p5"}


def test_no_sensitivities_returns_all_products(
    mixed_catalog: list[Product],
    standard_constraints: UserConstraints,
) -> None:
    result = filter_safe_products(mixed_catalog, standard_constraints)
    assert len(result) == len(mixed_catalog)


def test_product_flagged_if_any_single_ingredient_matches(
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    ingredients = [f"ingredient_{i}" for i in range(9)] + ["fragrance"]
    products = [
        Product(id="ten-ingredient-serum", name="Ten Ingredient Serum",
                category=ProductCategory.SERUM,
                ingredients=ingredients, description=""),
    ]
    result = filter_safe_products(products, fragrance_sensitive_constraints)
    assert result == []


# ---------------------------------------------------------------------------
# Sensitivity synonym expansion
# ---------------------------------------------------------------------------


def test_expand_fragrance_includes_parfum() -> None:
    assert "parfum" in _expand_sensitivities(["fragrance"])


def test_expand_fragrance_includes_perfume() -> None:
    assert "perfume" in _expand_sensitivities(["fragrance"])


def test_expand_fragrance_includes_fragrance() -> None:
    assert "fragrance" in _expand_sensitivities(["fragrance"])


def test_expand_alcohol_includes_alcohol_denat() -> None:
    expanded = _expand_sensitivities(["alcohol"])
    assert "alcohol denat" in expanded
    assert "alcohol" in expanded


def test_expand_unknown_passes_through() -> None:
    assert _expand_sensitivities(["latex"]) == {"latex"}


def test_expand_multiple_sensitivities() -> None:
    expanded = _expand_sensitivities(["fragrance", "alcohol"])
    assert "parfum" in expanded
    assert "alcohol denat" in expanded


# ---------------------------------------------------------------------------
# Sensitivity filtering with synonyms
# ---------------------------------------------------------------------------


def test_fragrance_sensitivity_blocks_parfum() -> None:
    constraints = UserConstraints(
        request_id="req-syn", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="p-parfum", name="Parfum Product",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "parfum", "glycerin"], description=""),
        Product(id="p-clean", name="Clean Product",
                category=ProductCategory.CLEANSER,
                ingredients=["water", "glycerin"], description=""),
    ]
    result = filter_safe_products(products, constraints)
    assert len(result) == 1
    assert result[0].id == "p-clean"


def test_fragrance_sensitivity_blocks_perfume() -> None:
    constraints = UserConstraints(
        request_id="req-syn", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="p-perfume", name="Perfume Product",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "perfume"], description=""),
    ]
    assert filter_safe_products(products, constraints) == []


def test_fragrance_sensitivity_blocks_fragrance_literal() -> None:
    constraints = UserConstraints(
        request_id="req-syn", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="p-frag", name="Frag Product",
                category=ProductCategory.SERUM,
                ingredients=["water", "fragrance"], description=""),
    ]
    assert filter_safe_products(products, constraints) == []


def test_alcohol_sensitivity_blocks_alcohol_denat() -> None:
    constraints = UserConstraints(
        request_id="req-alc", sensitivities=["alcohol"], max_products=5,
    )
    products = [
        Product(id="p-denat", name="Denat Product",
                category=ProductCategory.TONER,
                ingredients=["water", "alcohol denat"], description=""),
    ]
    assert filter_safe_products(products, constraints) == []


def test_alcohol_sensitivity_does_not_block_cetyl_alcohol() -> None:
    constraints = UserConstraints(
        request_id="req-fatty", sensitivities=["alcohol"], max_products=5,
    )
    products = [
        Product(id="p-cetyl", name="Cetyl Product",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "cetyl alcohol"], description=""),
    ]
    result = filter_safe_products(products, constraints)
    assert len(result) == 1
    assert result[0].id == "p-cetyl"


def test_safe_products_pass_through_with_sensitivities() -> None:
    constraints = UserConstraints(
        request_id="req-safe", sensitivities=["fragrance", "alcohol"],
        max_products=5,
    )
    products = [
        Product(id="p-safe", name="Safe Product", category=ProductCategory.SPF,
                ingredients=["water", "zinc oxide", "niacinamide"], description=""),
    ]
    assert len(filter_safe_products(products, constraints)) == 1


# ---------------------------------------------------------------------------
# Flag-based sensitivity filtering
# ---------------------------------------------------------------------------


def test_fragrance_flag_blocks_even_without_ingredient_match() -> None:
    constraints = UserConstraints(
        request_id="req-flag", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="exotic-oil", name="Botanical Cream",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "rosa damascena flower oil"],
                description="", contains_fragrance=True),
        Product(id="clean", name="Clean Cream",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "glycerin"],
                description="", contains_fragrance=False),
    ]
    result = filter_safe_products(products, constraints)
    assert len(result) == 1
    assert result[0].id == "clean"


def test_alcohol_flag_blocks_product() -> None:
    constraints = UserConstraints(
        request_id="req-alcflag", sensitivities=["alcohol"], max_products=5,
    )
    products = [
        Product(id="alc-flagged", name="Alcohol Product",
                category=ProductCategory.TONER,
                ingredients=["water", "isopropyl alcohol"],
                description="", contains_alcohol=True),
    ]
    assert filter_safe_products(products, constraints) == []


def test_fragrance_flag_false_passes_through() -> None:
    constraints = UserConstraints(
        request_id="req-ok", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="safe", name="Unscented SPF",
                category=ProductCategory.SPF,
                ingredients=["water", "zinc oxide"],
                description="", contains_fragrance=False),
    ]
    assert len(filter_safe_products(products, constraints)) == 1


def test_fragrance_flag_and_ingredient_both_caught() -> None:
    constraints = UserConstraints(
        request_id="req-both", sensitivities=["fragrance"], max_products=5,
    )
    products = [
        Product(id="double", name="Scented Product",
                category=ProductCategory.SERUM,
                ingredients=["water", "parfum"],
                description="", contains_fragrance=True),
    ]
    assert filter_safe_products(products, constraints) == []


# ---------------------------------------------------------------------------
# score_product_fit — deterministic fit scoring
# ---------------------------------------------------------------------------


def test_fit_oily_product_scores_high_for_oily_conditions() -> None:
    product = Product(
        id="oily-c", name="Oily Skin Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Lightweight gel cleanser for oily skin",
        skin_types=["oily", "combination"],
        concerns=["oiliness", "pores"],
        benefits=["oil control", "pore cleansing"],
    )
    score = score_product_fit(product, ["oily"])
    assert score > 4.0


def test_fit_dry_product_scores_negative_for_oily_conditions() -> None:
    product = Product(
        id="dry-c", name="Rich Cream Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Rich cream for dry skin",
        skin_types=["dry", "sensitive"],
        concerns=["dryness", "dehydration"],
        benefits=["hydrating", "nourishing"],
    )
    score = score_product_fit(product, ["oily", "acne"])
    assert score < 0


def test_fit_direct_match_beats_compatible_match() -> None:
    direct = Product(
        id="direct", name="Oily Moisturizer",
        category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="",
        skin_types=["oily"],
    )
    compatible = Product(
        id="compat", name="Combo Moisturizer",
        category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="",
        skin_types=["combination"],
    )
    score_direct = score_product_fit(direct, ["oily"])
    score_compat = score_product_fit(compatible, ["oily"])
    assert score_direct > score_compat


def test_fit_concerns_boost_score() -> None:
    with_concerns = Product(
        id="wc", name="P1", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        skin_types=["oily"],
        concerns=["acne", "oiliness", "pores"],
    )
    without_concerns = Product(
        id="nc", name="P2", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        skin_types=["oily"],
    )
    assert score_product_fit(with_concerns, ["oily", "acne"]) > \
           score_product_fit(without_concerns, ["oily", "acne"])


def test_fit_benefits_boost_score() -> None:
    with_benefits = Product(
        id="wb", name="P1", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="",
        skin_types=["dry"],
        benefits=["hydrating", "nourishing", "barrier support"],
    )
    without_benefits = Product(
        id="nb", name="P2", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="",
        skin_types=["dry"],
    )
    assert score_product_fit(with_benefits, ["dry"]) > \
           score_product_fit(without_benefits, ["dry"])


def test_fit_no_conditions_returns_near_zero() -> None:
    product = Product(
        id="p", name="P", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        skin_types=["oily"],
    )
    assert score_product_fit(product, []) == pytest.approx(0.0, abs=0.5)


def test_fit_fragrance_penalty_stronger_on_leave_on_categories() -> None:
    """Fragranced moisturizer scores lower than fragrance-free twin (tiered penalty)."""
    base = dict(
        ingredients=["water", "niacinamide"], description="",
        skin_types=["oily"], concerns=["oiliness"], benefits=["hydrating"],
    )
    fragrance_free = Product(
        id="m-ff", name="FF Moisturizer", category=ProductCategory.MOISTURIZER,
        contains_fragrance=False, **base,
    )
    fragranced = Product(
        id="m-sc", name="Scented Moisturizer", category=ProductCategory.MOISTURIZER,
        contains_fragrance=True, **base,
    )
    diff = (
        score_product_fit(fragrance_free, ["oily"])
        - score_product_fit(fragranced, ["oily"])
    )
    assert diff == pytest.approx(1.0, abs=0.01)


def test_fit_fragrance_penalty_smaller_for_cleanser_than_moisturizer() -> None:
    """Cleanser fragrance soft penalty is smaller than leave-on (moisturizer)."""
    c_base = dict(
        category=ProductCategory.CLEANSER,
        ingredients=["water", "glycerin"], description="",
        skin_types=["oily"], concerns=["oiliness"],
    )
    cle_ff = Product(
        id="c-ff", name="FF Cleanser", contains_fragrance=False, **c_base,
    )
    cle_sc = Product(
        id="c-sc", name="Scented Cleanser", contains_fragrance=True, **c_base,
    )
    m_base = dict(
        category=ProductCategory.MOISTURIZER,
        ingredients=["water", "glycerin"], description="",
        skin_types=["oily"], concerns=["oiliness"],
    )
    m_ff = Product(
        id="m-ff", name="FF Moist", contains_fragrance=False, **m_base,
    )
    m_sc = Product(
        id="m-sc", name="Scented Moist", contains_fragrance=True, **m_base,
    )
    d_cle = score_product_fit(cle_ff, ["oily"]) - score_product_fit(cle_sc, ["oily"])
    d_moi = score_product_fit(m_ff, ["oily"]) - score_product_fit(m_sc, ["oily"])
    assert d_moi > d_cle
    assert d_cle == pytest.approx(0.82, abs=0.01)
    assert d_moi == pytest.approx(1.0, abs=0.01)


def test_fit_fragrance_penalty_stronger_on_oily_acne_cleanser() -> None:
    """Oily+acne fragranced cleanser uses a stronger wash-off penalty than default."""
    base = dict(
        category=ProductCategory.CLEANSER,
        ingredients=["water", "glycerin"], description="",
        skin_types=["oily"], concerns=["oiliness"],
    )
    ff = Product(id="c-ff", name="FF", contains_fragrance=False, **base)
    sc = Product(id="c-sc", name="Sc", contains_fragrance=True, **base)
    diff = score_product_fit(ff, ["oily", "acne"]) - score_product_fit(sc, ["oily", "acne"])
    assert diff == pytest.approx(1.12, abs=0.01)


def test_fit_fragrance_penalty_stronger_on_dry_acne_cleanser() -> None:
    """Dry+acne (barrier-sensitive acne) uses the same stronger wash-off fragrance tier."""
    base = dict(
        category=ProductCategory.CLEANSER,
        ingredients=["water", "glycerin"], description="",
        skin_types=["dry"], concerns=["dryness"],
    )
    ff = Product(id="c-ff-d", name="FF", contains_fragrance=False, **base)
    sc = Product(id="c-sc-d", name="Sc", contains_fragrance=True, **base)
    diff = score_product_fit(ff, ["dry", "acne"]) - score_product_fit(sc, ["dry", "acne"])
    assert diff == pytest.approx(1.12, abs=0.01)


def test_fit_acne_only_keeps_default_cleanser_fragrance_penalty() -> None:
    """Acne without oily/barrier-sensitive labels keeps the default cleanser fragrance gap."""
    base = dict(
        category=ProductCategory.CLEANSER,
        ingredients=["water", "glycerin"], description="",
        skin_types=["combination"], concerns=["acne"],
    )
    ff = Product(id="c-ff-a", name="FF", contains_fragrance=False, **base)
    sc = Product(id="c-sc-a", name="Sc", contains_fragrance=True, **base)
    diff = score_product_fit(ff, ["acne"]) - score_product_fit(sc, ["acne"])
    assert diff == pytest.approx(0.82, abs=0.01)


def test_fit_barrier_damage_label_triggers_acne_ff_gentle_cleanser_fragrance_tier() -> None:
    """Underscore or phrase barrier labels still trigger the stronger cleanser tier with acne."""
    base = dict(
        category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        skin_types=["combination"],
    )
    ff = Product(id="c-ff-b", name="FF", contains_fragrance=False, **base)
    sc = Product(id="c-sc-b", name="Sc", contains_fragrance=True, **base)
    diff = score_product_fit(ff, ["acne", "barrier_damage"]) - score_product_fit(
        sc, ["acne", "barrier_damage"],
    )
    assert diff == pytest.approx(1.12, abs=0.01)


def test_fit_cleanser_sa_extra_deduction_when_routine_has_overlapping_acne_actives() -> None:
    """Optional hook: salicylic cleanser loses more fit when SA/BPO already in routine."""
    p = Product(
        id="sa-cle", name="SA Wash", category=ProductCategory.CLEANSER,
        ingredients=["water", "salicylic acid"], description="",
        skin_types=["oily"], concerns=["acne"],
    )
    base = score_product_fit(p, ["oily", "acne"])
    with_overlap = score_product_fit(
        p,
        ["oily", "acne"],
        routine_treatment_families=frozenset({"salicylic_acid"}),
    )
    assert base - with_overlap == pytest.approx(0.28, abs=0.01)
    base_dry = score_product_fit(p, ["dry", "acne"])
    with_overlap_dry = score_product_fit(
        p,
        ["dry", "acne"],
        routine_treatment_families=frozenset({"benzoyl_peroxide"}),
    )
    assert base_dry - with_overlap_dry == pytest.approx(0.28, abs=0.01)


def test_pick_prefers_fragrance_free_cleanser_on_tied_fit() -> None:
    """When conflict-adjusted fit ties, fragrance-free wins (cleanser slot)."""
    frag_first = Product(
        id="c-sc", name="Scented", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=True,
    )
    frag_free = Product(
        id="c-ff", name="FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
    )
    fit = {frag_first.id: 4.0, frag_free.id: 4.0}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [frag_first, frag_free],
        [],
        fit,
        force=False,
    )
    assert chosen is not None
    assert chosen.id == "c-ff"


def test_pick_prefers_fragrance_free_cleanser_when_within_epsilon_of_top() -> None:
    """Fragranced cleanser leads on raw fit but is within epsilon → pick FF."""
    scented = Product(
        id="c-sc", name="Scented", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=True,
    )
    frag_free = Product(
        id="c-ff", name="FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
    )
    # t_max=7.2, floor=6.75 — both in band; FF wins on fragrance sort key.
    fit = {scented.id: 7.2, frag_free.id: 7.0}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [scented, frag_free],
        [],
        fit,
        force=False,
    )
    assert chosen is not None
    assert chosen.id == "c-ff"


def test_pick_keeps_fragranced_cleanser_when_fit_leads_by_more_than_epsilon() -> None:
    """Clear fit gap vs best → highest adjusted fit wins even if fragranced."""
    scented = Product(
        id="c-sc", name="Scented", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=True,
    )
    frag_free = Product(
        id="c-ff", name="FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
    )
    fit = {scented.id: 8.0, frag_free.id: 7.0}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [scented, frag_free],
        [],
        fit,
        force=False,
    )
    assert chosen is not None
    assert chosen.id == "c-sc"


def test_pick_oily_acne_wider_band_prefers_fragrance_free_on_close_scores() -> None:
    """Oily+acne uses a wider near-top epsilon so FF can win when raw fit trails slightly."""
    scented = Product(
        id="c-sc", name="Scented", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=True,
    )
    frag_free = Product(
        id="c-ff", name="FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
    )
    fit = {scented.id: 7.9, frag_free.id: 7.5}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [scented, frag_free],
        [],
        fit,
        force=False,
        skin_conditions=["oily", "acne"],
    )
    assert chosen is not None
    assert chosen.id == "c-ff"


def test_pick_oily_acne_among_ff_prefers_more_gentle_benefit_signals() -> None:
    """When both FF and conflict-adjusted fit tie, prefer richer gentle/soothing benefits."""
    plain = Product(
        id="c-plain", name="Plain FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
        benefits=["oil control"],
    )
    gentle = Product(
        id="c-gentle", name="Gentle FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
        benefits=["gentle cleansing", "soothing"],
    )
    fit = {plain.id: 5.0, gentle.id: 5.0}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [plain, gentle],
        [],
        fit,
        force=False,
        skin_conditions=["oily", "acne"],
    )
    assert chosen is not None
    assert chosen.id == "c-gentle"


def test_pick_dry_acne_wider_band_prefers_fragrance_free_on_close_scores() -> None:
    """Dry+acne uses the same wide near-top epsilon as oily+acne."""
    scented = Product(
        id="c-sc-d", name="Scented", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=True,
    )
    frag_free = Product(
        id="c-ff-d", name="FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
    )
    fit = {scented.id: 7.9, frag_free.id: 7.5}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [scented, frag_free],
        [],
        fit,
        force=False,
        skin_conditions=["dry", "acne"],
    )
    assert chosen is not None
    assert chosen.id == "c-ff-d"


def test_pick_acne_irritation_prefers_gentler_ff_on_tie() -> None:
    """Acne + irritation label triggers gentle-benefit tie-break among FF cleansers."""
    plain = Product(
        id="c-p", name="Plain FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
        benefits=["oil control"],
    )
    gentle = Product(
        id="c-g", name="Gentle FF", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", contains_fragrance=False,
        benefits=["gentle cleansing", "calming"],
    )
    fit = {plain.id: 5.0, gentle.id: 5.0}
    chosen = _pick_with_active_policy(
        ProductCategory.CLEANSER,
        [plain, gentle],
        [],
        fit,
        force=False,
        skin_conditions=["acne", "irritation"],
    )
    assert chosen is not None
    assert chosen.id == "c-g"


def test_fit_body_spf_penalised() -> None:
    body_spf = Product(
        id="body-spf", name="Body SPF 50",
        category=ProductCategory.SPF,
        ingredients=["water", "zinc oxide"],
        description="Broad-spectrum body sunscreen for daily use",
        skin_types=["oily", "dry", "combination", "sensitive", "normal"],
    )
    face_spf = Product(
        id="face-spf", name="Face SPF 50",
        category=ProductCategory.SPF,
        ingredients=["water", "zinc oxide"],
        description="Lightweight face sunscreen for daily use",
        skin_types=["oily", "combination"],
    )
    assert score_product_fit(body_spf, ["oily"]) < \
           score_product_fit(face_spf, ["oily"])


def test_fit_semantic_score_as_small_bonus() -> None:
    product = Product(
        id="p", name="Generic", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
    )
    base = score_product_fit(product, ["oily"], semantic_score=0.0)
    boosted = score_product_fit(product, ["oily"], semantic_score=1.0)
    assert boosted - base == pytest.approx(0.5, abs=0.01)


def test_fit_semantic_cannot_override_skin_type_mismatch() -> None:
    """Even a perfect semantic score cannot make a mismatched product
    outscore a well-matched one."""
    mismatched = Product(
        id="bad", name="Dry Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Rich cream for dry skin",
        skin_types=["dry", "sensitive"],
        concerns=["dryness"],
        benefits=["nourishing"],
    )
    matched = Product(
        id="good", name="Oily Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Lightweight gel for oily skin",
        skin_types=["oily", "combination"],
        concerns=["oiliness", "pores"],
        benefits=["oil control"],
    )
    score_bad = score_product_fit(mismatched, ["oily", "acne"], semantic_score=1.0)
    score_good = score_product_fit(matched, ["oily", "acne"], semantic_score=0.0)
    assert score_good > score_bad


def test_fit_oily_prefers_oily_targeted_over_dry_focused() -> None:
    oily_targeted = Product(
        id="oily-t", name="Niacinamide Gel", category=ProductCategory.SERUM,
        ingredients=["niacinamide", "water"],
        description="Lightweight oil control for oily skin and pores",
        skin_types=["oily"],
        concerns=["oiliness", "pores"],
        benefits=["oil control", "pore refining", "lightweight"],
    )
    dry_focused = Product(
        id="dry-f", name="Rich Barrier Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["ceramide", "squalane"],
        description="Rich cream especially suitable for dry skin intense moisture",
        skin_types=["dry"],
        concerns=["dryness"],
        benefits=["barrier repair", "nourishing"],
    )
    assert score_product_fit(oily_targeted, ["oily"]) > score_product_fit(
        dry_focused, ["oily"],
    )


def test_fit_dry_prefers_hydrating_barrier_over_oil_control() -> None:
    hydrating = Product(
        id="hyd", name="HA Ceramide Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["hyaluronic acid", "ceramides", "glycerin"],
        description="Barrier support hydrating non stripping for dry skin",
        skin_types=["dry", "sensitive"],
        concerns=["dryness", "barrier damage"],
        benefits=["hydrating", "barrier support", "soothing"],
    )
    oil_control = Product(
        id="oc", name="Mattifying Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["salicylic acid"],
        description="Oil control matte finish for oily skin",
        skin_types=["oily"],
        concerns=["oiliness"],
        benefits=["oil control", "matte finish"],
    )
    assert score_product_fit(hydrating, ["dry"]) > score_product_fit(
        oil_control, ["dry"],
    )


def test_fit_normal_prefers_balanced_over_extreme_oily_or_dry_only() -> None:
    balanced = Product(
        id="bal", name="Gentle Daily Lotion", category=ProductCategory.MOISTURIZER,
        ingredients=["glycerin", "panthenol"],
        description="Balanced gentle daily hydration for normal skin",
        skin_types=["normal", "combination"],
        concerns=["dehydration"],
        benefits=["gentle cleansing", "balanced hydration", "lightweight"],
    )
    oily_only = Product(
        id="oo", name="Heavy Oil Control", category=ProductCategory.MOISTURIZER,
        ingredients=["water"],
        description="Heavy oil control acne treatment mattifying",
        skin_types=["oily"],
        concerns=["oiliness", "acne"],
        benefits=["oil control", "matte finish"],
    )
    dry_only = Product(
        id="do", name="Very Dry Rich Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["ceramides"],
        description="Rich cream for dry skin especially suitable for very dry",
        skin_types=["dry"],
        concerns=["dryness"],
        benefits=["intense moisture", "nourishing"],
    )
    s_bal = score_product_fit(balanced, ["normal"])
    assert s_bal > score_product_fit(oily_only, ["normal"])
    assert s_bal > score_product_fit(dry_only, ["normal"])


def test_fit_acne_prefers_acne_oil_control_over_rich_dry() -> None:
    acne_serum = Product(
        id="ac", name="BHA Pore Serum", category=ProductCategory.SERUM,
        ingredients=["salicylic acid", "niacinamide"],
        description="Acne blemish pore cleansing oil control lightweight",
        skin_types=["oily", "combination"],
        concerns=["acne", "oiliness", "pores"],
        benefits=["acne support", "pore cleansing", "oil control"],
    )
    rich_dry = Product(
        id="rd", name="Nourishing Night Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["squalane"],
        description="Nourishing cream for dry skin intense moisture barrier",
        skin_types=["dry"],
        concerns=["dryness"],
        benefits=["nourishing", "deep hydration"],
    )
    assert score_product_fit(acne_serum, ["acne"]) > score_product_fit(
        rich_dry, ["acne"],
    )


def test_fit_broad_all_skin_types_does_not_beat_better_targeted() -> None:
    """Many listed types dilute skin weight; concern/benefit/ingredient fit breaks ties."""
    broad = Product(
        id="br", name="Universal Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water", "glycerin"],
        description="Daily moisturizer",
        skin_types=["oily", "dry", "combination", "normal", "sensitive"],
        concerns=["dehydration"],
        benefits=["hydrating"],
    )
    targeted = Product(
        id="tg", name="Oily Skin Fluid", category=ProductCategory.MOISTURIZER,
        ingredients=["niacinamide"],
        description="Oil control lightweight non-greasy for oily skin pores",
        skin_types=["oily", "combination"],
        concerns=["oiliness", "pores"],
        benefits=["oil control", "pore refining", "lightweight"],
    )
    assert score_product_fit(targeted, ["oily"]) > score_product_fit(broad, ["oily"])


# ---------------------------------------------------------------------------
# select_best_per_category
# ---------------------------------------------------------------------------


def test_best_per_category_oily_prefers_oily_cleanser() -> None:
    """REQ 1: oily + acne prefers an oily/acne-targeted cleanser."""
    dry_cleanser = Product(
        id="cle-dry", name="Dry Skin Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Especially suitable for dry and sensitive skin",
        skin_types=["dry", "sensitive"], concerns=["dryness"],
    )
    oily_cleanser = Product(
        id="cle-oily", name="Oily Skin Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Gentle cleanser for oily acne-prone skin",
        skin_types=["oily", "combination"], concerns=["acne", "oiliness"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["oily"],
    )
    spf = Product(
        id="spf1", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="", skin_types=["oily"],
    )
    result = select_best_per_category(
        [dry_cleanser, oily_cleanser, moisturizer, spf],
        ["oily", "acne"],
    )
    cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert len(cleanser) == 1
    assert cleanser[0].id == "cle-oily"


def test_best_per_category_dry_prefers_dry_products() -> None:
    """REQ 2: dry prefers a dry-supporting cleanser/SPF."""
    oily_cleanser = Product(
        id="cle-oily", name="Oil Control Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Mattifying cleanser for oily skin",
        skin_types=["oily"], concerns=["oiliness"], benefits=["oil control"],
    )
    dry_cleanser = Product(
        id="cle-dry", name="Gentle Hydrating Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Nourishing cleanser for dry skin",
        skin_types=["dry", "sensitive"], concerns=["dryness"], benefits=["hydrating"],
    )
    spf_oily = Product(
        id="spf-oily", name="Matte SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="Matte finish SPF for oily skin",
        skin_types=["oily"], benefits=["mattifying"],
    )
    spf_dry = Product(
        id="spf-dry", name="Hydrating SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="Hydrating SPF for dry skin",
        skin_types=["dry", "sensitive"], benefits=["hydrating"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["dry"],
    )
    result = select_best_per_category(
        [oily_cleanser, dry_cleanser, spf_oily, spf_dry, moisturizer],
        ["dry"],
    )
    ids = {p.id for p in result}
    assert "cle-dry" in ids
    assert "spf-dry" in ids


def test_best_per_category_body_spf_penalised() -> None:
    """REQ 3: body-focused SPF is penalised in a facial routine."""
    body_spf = Product(
        id="spf-body", name="Body SPF 50", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="Body sunscreen lotion",
        skin_types=["oily", "dry", "combination"],
    )
    face_spf = Product(
        id="spf-face", name="Face SPF 50", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="Lightweight face sunscreen",
        skin_types=["oily", "combination"],
    )
    cleanser = Product(
        id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", skin_types=["oily"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["oily"],
    )
    result = select_best_per_category(
        [body_spf, face_spf, cleanser, moisturizer],
        ["oily"],
    )
    spf = [p for p in result if p.category == ProductCategory.SPF]
    assert len(spf) == 1
    assert spf[0].id == "spf-face"


def test_best_per_category_weak_optional_excluded() -> None:
    """REQ 4: optional serum/toner excluded when fit score is weak."""
    cleanser = Product(
        id="c1", name="Oily Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="For oily skin",
        skin_types=["oily"], concerns=["oiliness"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="Lightweight moisturizer",
        skin_types=["oily"],
    )
    spf = Product(
        id="spf1", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="Face SPF",
        skin_types=["oily"],
    )
    weak_serum = Product(
        id="ser1", name="Dry Skin Serum", category=ProductCategory.SERUM,
        ingredients=["water"], description="Rich hydrating serum for dry skin",
        skin_types=["dry", "sensitive"], concerns=["dryness"],
    )
    weak_toner = Product(
        id="ton1", name="Dry Toner", category=ProductCategory.TONER,
        ingredients=["water"], description="Soothing toner for dry skin",
        skin_types=["dry"], concerns=["dryness"],
    )
    result = select_best_per_category(
        [cleanser, moisturizer, spf, weak_serum, weak_toner],
        ["oily", "acne"],
    )
    categories = {p.category for p in result}
    assert ProductCategory.SERUM not in categories
    assert ProductCategory.TONER not in categories
    assert len(result) == 3


def test_best_per_category_strong_optional_included() -> None:
    """A serum that strongly matches should be included."""
    cleanser = Product(
        id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="", skin_types=["oily"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["oily"],
    )
    spf = Product(
        id="spf1", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="", skin_types=["oily"],
    )
    strong_serum = Product(
        id="ser1", name="Niacinamide Serum", category=ProductCategory.SERUM,
        ingredients=["water", "niacinamide"],
        description="Lightweight serum for oily acne-prone skin",
        skin_types=["oily", "combination"],
        concerns=["acne", "oiliness", "pores"],
        benefits=["oil control", "pore cleansing"],
    )
    result = select_best_per_category(
        [cleanser, moisturizer, spf, strong_serum],
        ["oily", "acne"],
    )
    ids = {p.id for p in result}
    assert "ser1" in ids


def test_best_per_category_essentials_always_present() -> None:
    """All 3 essential categories should be present when candidates exist."""
    products = [
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = select_best_per_category(products, ["oily"])
    categories = {p.category for p in result}
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_best_per_category_output_sorted_by_priority() -> None:
    products = [
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
    ]
    result = select_best_per_category(products, ["oily"])
    assert [p.category for p in result] == [
        ProductCategory.CLEANSER, ProductCategory.MOISTURIZER, ProductCategory.SPF,
    ]


def test_best_per_category_empty_input() -> None:
    assert select_best_per_category([], ["oily"]) == []


# ---------------------------------------------------------------------------
# Active-conflict avoidance in select_best_per_category
# ---------------------------------------------------------------------------


def test_active_families_detects_azelaic_acid() -> None:
    product = Product(
        id="az1", name="AZ Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["water", "azelaic acid", "glycerin"], description="",
    )
    assert "azelaic_acid" in active_families(product)


def test_active_families_detects_retinoid() -> None:
    product = Product(
        id="ret1", name="Retinol Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["water", "retinol", "glycerin"], description="",
    )
    assert "retinoid" in active_families(product)


def test_active_families_empty_for_gentle_product() -> None:
    product = Product(
        id="gen1", name="Gentle Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["water", "glycerin", "shea butter"], description="",
    )
    assert active_families(product) == frozenset()


def test_active_families_detects_multiple() -> None:
    product = Product(
        id="multi1", name="Multi Active", category=ProductCategory.SERUM,
        ingredients=["water", "salicylic acid", "niacinamide", "glycolic acid"],
        description="",
    )
    families = active_families(product)
    assert "salicylic_acid" in families
    assert "glycolic_acid" in families


def test_duplicate_strong_active_avoided_across_categories() -> None:
    products = [
        Product(id="cle1", name="Gentle Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "glycerin"], description=""),
        Product(id="moi-az", name="AZ Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "azelaic acid"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="ser-az", name="AZ Serum", category=ProductCategory.SERUM,
                ingredients=["water", "azelaic acid"], description=""),
        Product(id="ser-gentle", name="Hydrating Serum", category=ProductCategory.SERUM,
                ingredients=["water", "hyaluronic acid"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    ids = {p.id for p in result}
    assert "moi-az" in ids
    assert "ser-az" not in ids
    assert "ser-gentle" in ids


def test_exfoliant_stacking_prevented() -> None:
    products = [
        Product(id="cle-sal", name="SA Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "salicylic acid"], description=""),
        Product(id="moi1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "glycerin"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="ton-gly", name="Glycolic Toner", category=ProductCategory.TONER,
                ingredients=["water", "glycolic acid"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    ids = {p.id for p in result}
    assert "cle-sal" in ids
    assert "ton-gly" not in ids


def test_exfoliant_stacking_gentle_toner_allowed() -> None:
    products = [
        Product(id="cle-sal", name="SA Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "salicylic acid"], description=""),
        Product(id="moi1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "glycerin"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="ton-gentle", name="Gentle Toner", category=ProductCategory.TONER,
                ingredients=["water", "witch hazel"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    ids = {p.id for p in result}
    assert "cle-sal" in ids
    assert "ton-gentle" in ids


def test_essential_with_conflict_still_selected() -> None:
    products = [
        Product(id="cle-sal", name="SA Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "salicylic acid"], description=""),
        Product(id="moi-sal", name="SA Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "salicylic acid"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    categories = {p.category for p in result}
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_essential_picks_non_conflicting_alternative() -> None:
    """Cleanser actives are not scored against moisturizer; first listed
    moisturizer wins when legacy path has no fit scores (tie-break by order)."""
    products = [
        Product(id="cle-sal", name="SA Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "salicylic acid"], description=""),
        Product(id="moi-gly", name="GA Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "glycolic acid"], description=""),
        Product(id="moi-gentle", name="Gentle Moisturizer",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "shea butter"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    ids = {p.id for p in result}
    assert "cle-sal" in ids
    assert "moi-gly" in ids
    assert "moi-gentle" not in ids


def test_retinoid_not_duplicated_across_products() -> None:
    products = [
        Product(id="cle1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="moi-ret", name="Retinol Cream", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "retinol"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="ser-ret", name="Retinol Serum", category=ProductCategory.SERUM,
                ingredients=["water", "retinol"], description=""),
        Product(id="ser-ha", name="HA Serum", category=ProductCategory.SERUM,
                ingredients=["water", "hyaluronic acid"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    ids = {p.id for p in result}
    assert "moi-ret" in ids
    assert "ser-ret" not in ids
    assert "ser-ha" in ids


# ---------------------------------------------------------------------------
# build_routine — legacy fallback
# ---------------------------------------------------------------------------


def test_routine_respects_max_products(
    mixed_catalog: list[Product],
    tight_max_constraints: UserConstraints,
) -> None:
    result = build_routine(mixed_catalog, tight_max_constraints)
    assert len(result) == 3


def test_priority_essentials_chosen_before_optionals(
    mixed_catalog: list[Product],
    tight_max_constraints: UserConstraints,
) -> None:
    result = build_routine(mixed_catalog, tight_max_constraints)
    categories = {p.category for p in result}
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_routine_never_exceeds_available_products(
    essential_catalog: list[Product],
    standard_constraints: UserConstraints,
) -> None:
    result = build_routine(essential_catalog, standard_constraints)
    assert len(result) == len(essential_catalog)


def test_empty_product_list_returns_empty(
    standard_constraints: UserConstraints,
) -> None:
    assert build_routine([], standard_constraints) == []


def test_invalid_max_products_raises_value_error() -> None:
    constraints = UserConstraints.model_construct(
        request_id="req-invalid", sensitivities=[], max_products=0,
    )
    with pytest.raises(ValueError):
        build_routine([], constraints)


# ---------------------------------------------------------------------------
# match_products — full pipeline (fit-scoring path)
# ---------------------------------------------------------------------------


def _fake_scorer(
    skin_conditions: list[str],
    products: list[Product],
) -> dict[str, float]:
    """Scorer that returns 0.5 for all products."""
    return {p.id: 0.5 for p in products}


def test_match_products_with_scorer_selects_best_per_category() -> None:
    constraints = UserConstraints(
        request_id="req-fit", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="cle-dry", name="Dry Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description="For dry skin",
                skin_types=["dry"], concerns=["dryness"]),
        Product(id="cle-oily", name="Oily Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description="For oily skin",
                skin_types=["oily"], concerns=["oiliness"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description="", skin_types=["oily"]),
    ]
    result = match_products(
        catalog=catalog,
        constraints=constraints,
        skin_conditions=["oily", "acne"],
        scorer=_fake_scorer,
    )
    cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert cleanser[0].id == "cle-oily"


def test_match_products_with_scorer_respects_max_products() -> None:
    constraints = UserConstraints(
        request_id="req-max", sensitivities=[], max_products=3,
    )
    catalog = [
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description="", skin_types=["oily"]),
        Product(id="ser1", name="Serum", category=ProductCategory.SERUM,
                ingredients=["water"], description="For oily skin",
                skin_types=["oily"], concerns=["oiliness"],
                benefits=["oil control"]),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["oily"], scorer=_fake_scorer,
    )
    assert len(result) <= 3


def test_match_products_filters_before_scoring(
    scented_catalog: list[Product],
    fragrance_sensitive_constraints: UserConstraints,
) -> None:
    """REQ 5: safety filtering still works with the new scoring pipeline."""
    seen_by_scorer: list[list[Product]] = []

    def spy_scorer(conds: list[str], prods: list[Product]) -> dict[str, float]:
        seen_by_scorer.append(prods)
        return {p.id: 0.5 for p in prods}

    result = match_products(
        catalog=scented_catalog,
        constraints=fragrance_sensitive_constraints,
        skin_conditions=["acne"],
        scorer=spy_scorer,
    )
    assert all(p.id != "scented-moisturizer" for p in seen_by_scorer[0])
    assert all(p.id != "scented-moisturizer" for p in result)


def test_match_products_without_scorer_falls_back(
    mixed_catalog: list[Product],
    tight_max_constraints: UserConstraints,
) -> None:
    result = match_products(catalog=mixed_catalog, constraints=tight_max_constraints)
    categories = {p.category for p in result}
    assert ProductCategory.CLEANSER in categories
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_match_products_empty_catalog(standard_constraints: UserConstraints) -> None:
    assert match_products(catalog=[], constraints=standard_constraints) == []


def test_match_products_invalid_max_products_raises() -> None:
    constraints = UserConstraints.model_construct(
        request_id="req-bad", sensitivities=[], max_products=0,
    )
    with pytest.raises(ValueError):
        match_products(catalog=[], constraints=constraints)


# ---------------------------------------------------------------------------
# End-to-end integration: fit scoring + safety + active conflicts
# ---------------------------------------------------------------------------


def test_e2e_oily_acne_selects_oily_cleanser_over_dry() -> None:
    """REQ 1 + 6: oily+acne user gets oily cleanser even if a bad
    semantic scorer prefers the dry one."""
    dry_cleanser = Product(
        id="cle-dry", name="Dry Skin Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Rich cleanser for dry and sensitive skin",
        skin_types=["dry", "sensitive"], concerns=["dryness"],
    )
    oily_cleanser = Product(
        id="cle-oily", name="Oily Cleanser", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="Gentle wash for oily acne-prone skin",
        skin_types=["oily", "combination"], concerns=["acne", "oiliness"],
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["oily", "combination"],
    )
    spf = Product(
        id="spf1", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="", skin_types=["oily"],
    )
    constraints = UserConstraints(
        request_id="req-e2e", sensitivities=[], max_products=5,
    )

    def bad_scorer(conds: list[str], prods: list[Product]) -> dict[str, float]:
        """Gives dry cleanser the best semantic score."""
        return {p.id: (0.95 if "dry" in p.id else 0.3) for p in prods}

    result = match_products(
        catalog=[dry_cleanser, oily_cleanser, moisturizer, spf],
        constraints=constraints,
        skin_conditions=["oily", "acne"],
        scorer=bad_scorer,
    )
    selected_cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert len(selected_cleanser) == 1
    assert selected_cleanser[0].id == "cle-oily"


def test_e2e_oily_acne_prefers_fragrance_free_gentle_cleanser_over_fragranced_sa() -> None:
    """Close-fit oily+acne routine: FF gentle wash beats fragranced salicylic cleanser."""
    sa_fr = Product(
        id="cle-sa-sc", name="SA Scented Wash", category=ProductCategory.CLEANSER,
        ingredients=["water", "salicylic acid", "niacinamide", "fragrance"],
        description="",
        skin_types=["oily", "combination"],
        concerns=["acne", "oiliness", "pores"],
        benefits=[
            "oil control", "pore cleansing", "lightweight", "non comedogenic",
        ],
        contains_fragrance=True,
    )
    ff_gentle = Product(
        id="cle-ff-g", name="Gentle FF Wash", category=ProductCategory.CLEANSER,
        ingredients=["water", "niacinamide", "glycerin"],
        description="",
        skin_types=["oily", "combination"],
        concerns=["acne", "oiliness", "pores"],
        benefits=[
            "gentle cleansing", "soothing", "oil control",
            "lightweight", "non comedogenic",
        ],
        contains_fragrance=False,
    )
    moisturizer = Product(
        id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["oily", "combination"],
    )
    spf = Product(
        id="spf1", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="", skin_types=["oily"],
    )
    constraints = UserConstraints(
        request_id="req-e2e-ff", sensitivities=[], max_products=5,
    )
    result = match_products(
        catalog=[sa_fr, ff_gentle, moisturizer, spf],
        constraints=constraints,
        skin_conditions=["oily", "acne"],
        scorer=lambda c, prods: {p.id: 0.5 for p in prods},
    )
    cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert len(cleanser) == 1
    assert cleanser[0].id == "cle-ff-g"


def test_e2e_dry_acne_prefers_fragrance_free_gentle_cleanser_over_fragranced_sa() -> None:
    """Dry+acne + barrier-sensitive scoring: FF gentle wash beats fragranced SA cleanser."""
    sa_fr = Product(
        id="cle-sa-dry", name="SA Scented Wash", category=ProductCategory.CLEANSER,
        ingredients=["water", "salicylic acid", "niacinamide", "fragrance"],
        description="",
        skin_types=["dry", "combination"],
        concerns=["acne", "dryness", "pores"],
        benefits=[
            "pore cleansing", "lightweight", "non comedogenic",
        ],
        contains_fragrance=True,
    )
    ff_gentle = Product(
        id="cle-ff-dry", name="Gentle FF Wash", category=ProductCategory.CLEANSER,
        ingredients=["water", "niacinamide", "glycerin"],
        description="",
        skin_types=["dry", "combination"],
        concerns=["acne", "dryness", "pores"],
        benefits=[
            "gentle cleansing", "soothing", "hydrating",
            "lightweight", "non comedogenic",
        ],
        contains_fragrance=False,
    )
    moisturizer = Product(
        id="m-dry", name="Moisturizer", category=ProductCategory.MOISTURIZER,
        ingredients=["water"], description="", skin_types=["dry", "combination"],
    )
    spf = Product(
        id="spf-dry", name="SPF", category=ProductCategory.SPF,
        ingredients=["zinc oxide"], description="", skin_types=["dry"],
    )
    constraints = UserConstraints(
        request_id="req-e2e-dry-acne", sensitivities=[], max_products=5,
    )
    result = match_products(
        catalog=[sa_fr, ff_gentle, moisturizer, spf],
        constraints=constraints,
        skin_conditions=["dry", "acne"],
        scorer=lambda c, prods: {p.id: 0.5 for p in prods},
    )
    cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert len(cleanser) == 1
    assert cleanser[0].id == "cle-ff-dry"


def test_e2e_dry_selects_dry_over_oily_products() -> None:
    """REQ 2: dry user gets dry-targeted products."""
    constraints = UserConstraints(
        request_id="req-dry", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="cle-oily", name="Oil Control Cleanser",
                category=ProductCategory.CLEANSER,
                ingredients=["water"], description="Mattifying for oily skin",
                skin_types=["oily"], concerns=["oiliness"], benefits=["oil control"]),
        Product(id="cle-dry", name="Hydrating Cleanser",
                category=ProductCategory.CLEANSER,
                ingredients=["water"], description="Gentle cleanser for dry skin",
                skin_types=["dry", "sensitive"], concerns=["dryness"],
                benefits=["hydrating", "gentle cleansing"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description="", skin_types=["dry"]),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description="", skin_types=["dry"]),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["dry"], scorer=_fake_scorer,
    )
    cleanser = [p for p in result if p.category == ProductCategory.CLEANSER]
    assert cleanser[0].id == "cle-dry"


def test_e2e_body_spf_avoided_for_facial_routine() -> None:
    """REQ 3: body-focused SPF is penalised."""
    constraints = UserConstraints(
        request_id="req-spf", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="spf-body", name="Body SPF 50", category=ProductCategory.SPF,
                ingredients=["zinc oxide"],
                description="Broad spectrum body sunscreen",
                skin_types=["oily", "dry", "combination"]),
        Product(id="spf-face", name="Face SPF 50", category=ProductCategory.SPF,
                ingredients=["zinc oxide"],
                description="Lightweight face sunscreen",
                skin_types=["oily", "combination"]),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["oily"], scorer=_fake_scorer,
    )
    spf = [p for p in result if p.category == ProductCategory.SPF]
    assert spf[0].id == "spf-face"


def test_e2e_weak_optional_excluded() -> None:
    """REQ 4: optional categories excluded when weakly matched."""
    constraints = UserConstraints(
        request_id="req-opt", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="c1", name="Oily Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description="For oily skin",
                skin_types=["oily"], concerns=["oiliness"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description="", skin_types=["oily"]),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description="", skin_types=["oily"]),
        Product(id="ser-dry", name="Dry Serum", category=ProductCategory.SERUM,
                ingredients=["water"],
                description="Rich hydrating serum for dry skin",
                skin_types=["dry", "sensitive"], concerns=["dryness"]),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["oily", "acne"], scorer=_fake_scorer,
    )
    ids = {p.id for p in result}
    assert "ser-dry" not in ids
    assert len(result) == 3


def test_e2e_safety_filtering_with_scorer() -> None:
    """REQ 5: safety filtering works with the new pipeline."""
    constraints = UserConstraints(
        request_id="req-safe", sensitivities=["fragrance"], max_products=5,
    )
    catalog = [
        Product(id="c1", name="Safe Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m-fragrance", name="Scented Moisturizer",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "parfum"], description=""),
        Product(id="m-safe", name="Safe Moisturizer",
                category=ProductCategory.MOISTURIZER,
                ingredients=["water", "glycerin"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["oily"], scorer=_fake_scorer,
    )
    ids = {p.id for p in result}
    assert "m-fragrance" not in ids
    assert "m-safe" in ids


def test_e2e_active_stacking_prevented_with_fit_scoring() -> None:
    """Active conflicts are still prevented in the new pipeline."""
    constraints = UserConstraints(
        request_id="req-acid", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="cle-sal", name="SA Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water", "salicylic acid"],
                description="Cleanser for acne-prone skin",
                skin_types=["oily"], concerns=["acne"]),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water", "niacinamide"], description="",
                skin_types=["oily"]),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description="",
                skin_types=["oily"]),
        Product(id="ton-gly", name="Glycolic Toner", category=ProductCategory.TONER,
                ingredients=["water", "glycolic acid"],
                description="Toner for oily skin",
                skin_types=["oily"], concerns=["oiliness"],
                benefits=["oil control"]),
        Product(id="ton-gentle", name="Gentle Toner", category=ProductCategory.TONER,
                ingredients=["water", "panthenol"],
                description="Gentle toner for oily skin",
                skin_types=["oily"], concerns=["oiliness"]),
    ]
    result = match_products(
        catalog=catalog, constraints=constraints,
        skin_conditions=["oily", "acne"], scorer=_fake_scorer,
    )
    ids = {p.id for p in result}
    assert "cle-sal" in ids
    assert "ton-gly" not in ids


# ---------------------------------------------------------------------------
# select_balanced_routine — legacy path
# ---------------------------------------------------------------------------


def test_balanced_routine_picks_one_per_category() -> None:
    products = [
        Product(id="c1", name="Cleanser A", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c2", name="Cleanser B", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c3", name="Cleanser C", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = select_balanced_routine(products, max_products=5)
    categories = [p.category for p in result]
    assert categories.count(ProductCategory.CLEANSER) == 1
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_balanced_routine_essentials_before_optionals() -> None:
    ranked = [
        Product(id="ton1", name="Toner", category=ProductCategory.TONER,
                ingredients=["water"], description=""),
        Product(id="ser1", name="Serum", category=ProductCategory.SERUM,
                ingredients=["water"], description=""),
        Product(id="oth1", name="Other", category=ProductCategory.OTHER,
                ingredients=["water"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="cle1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="moi1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
    ]
    result = select_balanced_routine(ranked, max_products=3)
    categories = {p.category for p in result}
    assert categories == {
        ProductCategory.CLEANSER, ProductCategory.MOISTURIZER, ProductCategory.SPF,
    }


def test_balanced_routine_output_sorted_by_priority() -> None:
    ranked = [
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="cle1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="moi1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
    ]
    result = select_balanced_routine(ranked, max_products=5)
    assert [p.category for p in result] == [
        ProductCategory.CLEANSER, ProductCategory.MOISTURIZER, ProductCategory.SPF,
    ]


def test_balanced_routine_empty_input() -> None:
    assert select_balanced_routine([], max_products=5) == []


def test_many_cleansers_produces_diverse_routine() -> None:
    constraints = UserConstraints(
        request_id="req-diverse", sensitivities=[], max_products=5,
    )
    catalog = [
        Product(id="c1", name="Cleanser 1", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c2", name="Cleanser 2", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c3", name="Cleanser 3", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c4", name="Cleanser 4", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
        Product(id="ser1", name="Serum", category=ProductCategory.SERUM,
                ingredients=["water"], description=""),
    ]
    result = match_products(catalog=catalog, constraints=constraints)
    categories = [p.category for p in result]
    assert categories.count(ProductCategory.CLEANSER) == 1
    assert ProductCategory.MOISTURIZER in categories
    assert ProductCategory.SPF in categories


def test_spf_included_when_optionals_rank_higher() -> None:
    constraints = UserConstraints(
        request_id="req-spf", sensitivities=[], max_products=3,
    )
    catalog = [
        Product(id="ser1", name="Serum", category=ProductCategory.SERUM,
                ingredients=["water"], description=""),
        Product(id="ton1", name="Toner", category=ProductCategory.TONER,
                ingredients=["water"], description=""),
        Product(id="oth1", name="Other", category=ProductCategory.OTHER,
                ingredients=["water"], description=""),
        Product(id="spf1", name="SPF", category=ProductCategory.SPF,
                ingredients=["zinc oxide"], description=""),
    ]
    result = match_products(catalog=catalog, constraints=constraints)
    categories = {p.category for p in result}
    assert ProductCategory.SPF in categories


# ---------------------------------------------------------------------------
# build_routine_rationale
# ---------------------------------------------------------------------------


def test_rationale_pipeline_steps_present() -> None:
    catalog = [
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
    ]
    rationale = build_routine_rationale(
        catalog=catalog, safe_products=catalog, selected=catalog,
        skin_conditions=["oily"], used_semantic_ranking=True,
    )
    step_names = [s["step"] for s in rationale["pipeline_steps"]]
    assert "safety_filter" in step_names
    assert "semantic_ranking" in step_names
    assert "routine_assembly" in step_names


def test_rationale_safety_filter_counts() -> None:
    catalog = [
        Product(id="c1", name="A", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c2", name="B", category=ProductCategory.CLEANSER,
                ingredients=["fragrance"], description=""),
    ]
    safe = [catalog[0]]
    selected = [catalog[0]]
    rationale = build_routine_rationale(
        catalog=catalog, safe_products=safe, selected=selected,
    )
    safety = rationale["pipeline_steps"][0]
    assert safety["catalog_size"] == 2
    assert safety["passed"] == 1
    assert safety["removed"] == 1


def test_rationale_uses_category_priority_when_no_semantic() -> None:
    product = Product(id="c1", name="A", category=ProductCategory.CLEANSER,
                      ingredients=["water"], description="")
    rationale = build_routine_rationale(
        catalog=[product], safe_products=[product], selected=[product],
        used_semantic_ranking=False,
    )
    step_names = [s["step"] for s in rationale["pipeline_steps"]]
    assert "category_priority" in step_names
    assert "semantic_ranking" not in step_names


def test_rationale_product_rationales_per_product() -> None:
    cleanser = Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                       ingredients=["water"], description="")
    serum = Product(id="s1", name="Serum", category=ProductCategory.SERUM,
                    ingredients=["water"], description="")
    rationale = build_routine_rationale(
        catalog=[cleanser, serum], safe_products=[cleanser, serum],
        selected=[cleanser, serum],
    )
    pr = rationale["product_rationales"]
    assert "c1" in pr
    assert pr["c1"]["role"] == "essential"
    assert pr["c1"]["category"] == "CLEANSER"
    assert "s1" in pr
    assert pr["s1"]["role"] == "optional"


def test_rationale_reports_strong_actives() -> None:
    product = Product(id="az1", name="AZ Cream", category=ProductCategory.MOISTURIZER,
                      ingredients=["water", "azelaic acid"], description="")
    rationale = build_routine_rationale(
        catalog=[product], safe_products=[product], selected=[product],
    )
    assert "azelaic_acid" in rationale["product_rationales"]["az1"]["strong_actives"]


def test_rationale_fragrance_penalty_tier_on_selected() -> None:
    moist = Product(
        id="m1", name="Scented Cream", category=ProductCategory.MOISTURIZER,
        ingredients=["water", "glycerin"], description="",
        contains_fragrance=True,
    )
    cle = Product(
        id="c1", name="Scented Wash", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        contains_fragrance=True,
    )
    rationale = build_routine_rationale(
        catalog=[moist, cle], safe_products=[moist, cle],
        selected=[moist, cle], skin_conditions=["oily"],
    )
    pr = rationale["product_rationales"]
    assert pr["m1"]["fragrance_ranking_penalty_tier"] == "leave_on"
    assert pr["c1"]["fragrance_ranking_penalty_tier"] == "cleanser"


def test_rationale_fragrance_tier_cleanser_oily_acne() -> None:
    cle = Product(
        id="c1", name="Scented Wash", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        contains_fragrance=True,
    )
    rationale = build_routine_rationale(
        catalog=[cle], safe_products=[cle], selected=[cle],
        skin_conditions=["oily", "acne"],
    )
    assert (
        rationale["product_rationales"]["c1"]["fragrance_ranking_penalty_tier"]
        == "cleanser_oily_acne"
    )


def test_rationale_fragrance_tier_cleanser_acne_barrier_sensitive() -> None:
    cle = Product(
        id="c1", name="Scented Wash", category=ProductCategory.CLEANSER,
        ingredients=["water"], description="",
        contains_fragrance=True,
    )
    rationale = build_routine_rationale(
        catalog=[cle], safe_products=[cle], selected=[cle],
        skin_conditions=["dry", "acne"],
    )
    assert (
        rationale["product_rationales"]["c1"]["fragrance_ranking_penalty_tier"]
        == "cleanser_acne_barrier_sensitive"
    )


def test_rationale_active_avoidances_listed() -> None:
    cleanser = Product(id="cle-sal", name="SA Cleanser",
                       category=ProductCategory.CLEANSER,
                       ingredients=["water", "salicylic acid"], description="")
    toner_gly = Product(id="ton-gly", name="Glycolic Toner",
                        category=ProductCategory.TONER,
                        ingredients=["water", "glycolic acid"], description="")
    rationale = build_routine_rationale(
        catalog=[cleanser, toner_gly], safe_products=[cleanser, toner_gly],
        selected=[cleanser],
    )
    assert "active_avoidances" in rationale
    avoidance_text = " ".join(rationale["active_avoidances"])
    assert "Glycolic Toner" in avoidance_text


def test_rationale_no_avoidances_when_no_conflicts() -> None:
    products = [
        Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["glycerin"], description=""),
    ]
    rationale = build_routine_rationale(
        catalog=products, safe_products=products, selected=products,
    )
    assert "active_avoidances" not in rationale


def test_rationale_candidates_in_category_count() -> None:
    products = [
        Product(id="c1", name="Cleanser A", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="c2", name="Cleanser B", category=ProductCategory.CLEANSER,
                ingredients=["water"], description=""),
        Product(id="m1", name="Moisturizer", category=ProductCategory.MOISTURIZER,
                ingredients=["water"], description=""),
    ]
    selected = [products[0], products[2]]
    rationale = build_routine_rationale(
        catalog=products, safe_products=products, selected=selected,
    )
    assert rationale["product_rationales"]["c1"]["candidates_in_category"] == 2
    assert rationale["product_rationales"]["m1"]["candidates_in_category"] == 1


def test_rationale_is_json_serializable() -> None:
    product = Product(id="c1", name="Cleanser", category=ProductCategory.CLEANSER,
                      ingredients=["water", "salicylic acid"], description="")
    rationale = build_routine_rationale(
        catalog=[product], safe_products=[product], selected=[product],
        skin_conditions=["oily", "acne"], used_semantic_ranking=True,
    )
    serialized = json.dumps(rationale)
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert "pipeline_steps" in parsed
