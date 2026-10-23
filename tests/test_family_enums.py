"""Family enums: the per-class views of IntegrationSchemeType.

The canonical enum does not tell a user which entry is an explicit
scheme and which is an implicit one, so the library also ships one enum per
driver family (enums.FAMILY_ENUMS). These tests pin the contract those views
must keep: exhaustive and disjoint over the pinned family sizes, same name and value
as the canonical members, `get_scheme` accepting every lookup form, and
`IntegrationScheme.family` reporting the right class.
"""

import pytest

from warpSPHIntegrators.schemes import CoupledRK, DIRK, Exponential, ExplicitRK, FAMILY_ENUMS, IMEX, IntegrationSchemeType, LinearlyImplicit, MultistepExplicit, MultistepImplicit, Newmark, RelaxedChebyshev, SCHEME_FAMILY, Symplectic
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.schemes import IntegrationSchemes

# The member count per family, in FAMILY_ENUMS order. A count change means a
# scheme moved families or a new one was registered -- both worth a look.
#: Wave 7 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13): the eleven high-order tableaus
#: (Vern6-9, RKV76IIa, TanYam7, TsitPap8, DP8, Feagin10/12/14) join ExplicitRK,
#: 52 -> 63; the registry grows 97 -> 108.
#: Wave 8 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13; docs/research/integrators/notes-3-28.md): the 29 remaining
#: ODE v7.8.1 DIRK tableaus join DIRK, 7 -> 36; the registry grows 108 -> 137.
EXPECTED_FAMILY_SIZES = (63, 6, 3, 10, 37, 4, 8, 7, 1, 2, 1)
EXPECTED_TOTAL = sum(EXPECTED_FAMILY_SIZES)

IMPLICIT_FAMILIES = (DIRK, CoupledRK, MultistepImplicit, IMEX, LinearlyImplicit, Newmark)
EXPLICIT_FAMILIES = (ExplicitRK, Symplectic, RelaxedChebyshev, MultistepExplicit, Exponential)


def test_families_are_exhaustive_and_disjoint():
    members = list(IntegrationSchemeType)
    assert len(members) == EXPECTED_TOTAL
    seen = []
    for family in FAMILY_ENUMS:
        for member in family:
            seen.append(IntegrationSchemeType(member.value))
    assert len(seen) == len(set(seen)) == EXPECTED_TOTAL, "a member appears in more than one family"
    assert set(seen) == set(members), "a canonical member is missing from every family"
    assert len(FAMILY_ENUMS) == 11
    assert tuple(len(list(family)) for family in FAMILY_ENUMS) == EXPECTED_FAMILY_SIZES
    assert set(SCHEME_FAMILY) == set(members)


def test_family_members_match_the_canonical_enum():
    for family in FAMILY_ENUMS:
        for member in family:
            canonical = IntegrationSchemeType[member.name]
            assert canonical.value == member.value, (family.__name__, member.name)


def test_every_registered_scheme_has_a_family():
    assert len(IntegrationSchemes) == EXPECTED_TOTAL
    for scheme in IntegrationSchemes:
        assert scheme.family is SCHEME_FAMILY[scheme.identifier]
        assert scheme.family in FAMILY_ENUMS


def test_family_matches_the_implicit_metadata():
    for scheme in IntegrationSchemes:
        if scheme.family in IMPLICIT_FAMILIES:
            assert scheme.implicit, scheme.name
        else:
            assert scheme.family in EXPLICIT_FAMILIES, scheme.name
            assert not scheme.implicit, scheme.name


def test_getintegrator_accepts_all_lookup_forms():
    by_value = {scheme.identifier.value: scheme for scheme in IntegrationSchemes}
    for family in FAMILY_ENUMS:
        for member in family:
            scheme = by_value[member.value]
            assert get_scheme(scheme.name) is scheme
            assert get_scheme(scheme.identifier) is scheme
            assert get_scheme(scheme.identifier.name) is scheme
            assert get_scheme(member) is scheme
            assert get_scheme(member).identifier is scheme.identifier


def test_unknown_integrator_reports_close_matches():
    with pytest.raises(ValueError, match="Did you mean"):
        get_scheme('RK5')
    with pytest.raises(ValueError) as excinfo:
        get_scheme(42)
    # 42 is not an Enum, so no name suggestion is possible.
    assert "Did you mean" not in str(excinfo.value)


def test_family_membership_spot_checks():
    expectations = [
        ('RK4', ExplicitRK),
        ('SSPRK(10,4)', ExplicitRK),
        ('Dormand-Prince 5(4)', ExplicitRK),
        ('Velocity Verlet', Symplectic),
        ('Semi-Implicit Euler', Symplectic),
        ('RKC2', RelaxedChebyshev),
        ('Adams-Bashforth 2', MultistepExplicit),
        ('Adams-Bashforth-Moulton 4 (PECE)', MultistepExplicit),
        ('SSPRKMSVS32', MultistepExplicit),
        ('SSPRKMSVS43', MultistepExplicit),
        ('Trapezoidal (Crank-Nicolson)', DIRK),
        ('ESDIRK4(3)6L[2]SA', DIRK),
        ('Gauss-Legendre 2', CoupledRK),
        ('Radau IIA s=2', CoupledRK),
        ('BDF5', MultistepImplicit),
        ('Adams-Moulton 3 (implicit)', MultistepImplicit),
        ('IMEX Euler', IMEX),
        ('ARK3(2)4L[2]SA', IMEX),
        ('CNAB2', IMEX),
        ('ROS3P', LinearlyImplicit),
        ('ETD2RK', Exponential),
        ('EXPRB32', Exponential),
        ('Newmark', Newmark),
        # Wave 7 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13): the high-order tableaus
        # land in ExplicitRK, the rest of the registry's ExplicitRK coverage
        # unchanged.
        ('Vern6 6(5)', ExplicitRK),
        ('DP8', ExplicitRK),
        ('Feagin14', ExplicitRK),
        # Wave 8 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13; docs/research/integrators/notes-3-28.md): the 29
        # remaining ODE v7.8.1 DIRK tableaus land in DIRK.
        ('Kvaerno5', DIRK),
        ('ESDIRK6(5)9L[2]SA', DIRK),
        ('Cash4', DIRK),
        ('SFSDIRK8', DIRK),
    ]
    for name, family in expectations:
        assert get_scheme(name).family is family, name
