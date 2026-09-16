"""Family enums: the per-class views of IntegrationSchemeType.

The single 64-member enum does not tell a user which entry is an explicit
scheme and which is an implicit one, so the library also ships one enum per
driver family (enums.FAMILY_ENUMS). These tests pin the contract those views
must keep: exhaustive and disjoint over the 64 members, same name and value
as the canonical members, `getIntegrator` accepting every lookup form, and
`IntegrationScheme.family` reporting the right class.
"""

import pytest

from warpSPHIntegrators import (
    CoupledRK,
    DIRK,
    Exponential,
    ExplicitRK,
    FAMILY_ENUMS,
    IMEX,
    IntegrationSchemeType,
    LinearlyImplicit,
    MultistepExplicit,
    MultistepImplicit,
    Newmark,
    RelaxedChebyshev,
    SCHEME_FAMILY,
    Symplectic,
    getIntegrator,
    getIntegrationEnum,
)
from warpSPHIntegrators.integration import IntegrationSchemes

# The member count per family, in FAMILY_ENUMS order. A count change means a
# scheme moved families or a new one was registered -- both worth a look.
EXPECTED_FAMILY_SIZES = (21, 6, 3, 7, 7, 2, 8, 6, 1, 2, 1)

IMPLICIT_FAMILIES = (DIRK, CoupledRK, MultistepImplicit, IMEX, LinearlyImplicit, Newmark)
EXPLICIT_FAMILIES = (ExplicitRK, Symplectic, RelaxedChebyshev, MultistepExplicit, Exponential)


def test_families_are_exhaustive_and_disjoint():
    members = list(IntegrationSchemeType)
    assert len(members) == 64
    seen = []
    for family in FAMILY_ENUMS:
        for member in family:
            seen.append(IntegrationSchemeType(member.value))
    assert len(seen) == len(set(seen)) == 64, "a member appears in more than one family"
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
    assert len(IntegrationSchemes) == 64
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
            assert getIntegrator(scheme.name) is scheme
            assert getIntegrator(scheme.identifier) is scheme
            assert getIntegrator(scheme.identifier.name) is scheme
            assert getIntegrator(member) is scheme
            assert getIntegrationEnum(member) is scheme.identifier


def test_unknown_integrator_reports_close_matches():
    with pytest.raises(ValueError, match="Did you mean"):
        getIntegrator('RK5')
    with pytest.raises(ValueError) as excinfo:
        getIntegrator(42)
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
    ]
    for name, family in expectations:
        assert getIntegrator(name).family is family, name
