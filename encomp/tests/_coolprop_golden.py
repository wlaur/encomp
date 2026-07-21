"""Compact CoolProp 8.0.0 references generated before removing its Python binding."""

from dataclasses import dataclass
from typing import Literal

from encomp.coolprop import AssumedPhase, CName, Composition
from encomp.fluids import CProperty


@dataclass(frozen=True)
class GoldenCase:
    id: str
    kind: Literal["fluid", "humid_air", "water"]
    inputs: tuple[tuple[CProperty, float, str], ...]
    output: CProperty
    expected: float
    name: CName | None = None
    composition: Composition | None = None
    phase: AssumedPhase | None = None


GOLDEN_CASES = (
    GoldenCase(
        id="water-if97-pt",
        kind="water",
        inputs=(("P", 5e6, "Pa"), ("T", 400.0, "K")),
        output="DMASS",
        expected=939.9062482796301,
    ),
    GoldenCase(
        id="heos-pure-pt",
        kind="fluid",
        name="HEOS::CarbonDioxide",
        inputs=(("P", 5e6, "Pa"), ("T", 310.0, "K")),
        output="HMASS",
        # Fluid converts CoolProp's J/kg result to its preferred kJ/kg unit.
        expected=462.41253710627754,
    ),
    GoldenCase(
        id="incompressible-concentration",
        kind="fluid",
        name="INCOMP::MEG[0.5]",
        inputs=(("P", 2e5, "Pa"), ("T", 300.0, "K")),
        output="DMASS",
        expected=1061.1793077204613,
    ),
    GoldenCase(
        id="embedded-mixture",
        kind="fluid",
        name="HEOS::CarbonDioxide[0.7]&Oxygen[0.3]",
        inputs=(("P", 5e6, "Pa"), ("T", 300.0, "K")),
        output="DMASS",
        expected=97.49264109222536,
    ),
    GoldenCase(
        id="explicit-composition",
        kind="fluid",
        name="HEOS",
        composition={"CarbonDioxide": 0.6, "Oxygen": 0.4},
        inputs=(("P", 5e6, "Pa"), ("T", 300.0, "K")),
        output="DMASS",
        expected=91.2386351314185,
    ),
    GoldenCase(
        id="assumed-gas-phase",
        kind="fluid",
        name="HEOS::Water",
        phase="gas",
        inputs=(("P", 1e5, "Pa"), ("T", 500.0, "K")),
        output="DMASS",
        expected=0.435140075089223,
    ),
    GoldenCase(
        id="non-pt-input-pair",
        kind="fluid",
        name="HEOS::Water",
        inputs=(("P", 2e6, "Pa"), ("H", 749714.8321521956, "J/kg")),
        output="DMASS",
        expected=891.0411792241698,
    ),
    GoldenCase(
        id="humid-air",
        kind="humid_air",
        inputs=(("P", 101325.0, "Pa"), ("T", 300.0, "K"), ("R", 0.5, "")),
        output="W",
        expected=0.01109552970536823,
    ),
)
