"""Turn an action's arguments into what the 1.19 click paths actually type and click.

The model speaks in ids (``SWE_defense_act``, ``infantry_at``, a state name); the
game's UI takes a localised name typed into a search box, or a pixel. This step
runs before any input is sent and can refuse the whole action: a name that cannot
be typed, a technology whose grid cell is off the screen, a calibration that does
not exist yet. Refusing here is the cheap, honest outcome -- half a script, typed
into the wrong place, is not.

Derived values a script may use as ``{placeholders}``:

* ``set_national_focus``: ``{focus_name}`` -- an ASCII-only fragment of the focus's
  English name. The focus tree's search (ctrl+f) matches names and descriptions.
* ``queue_construction``: ``{state_name}`` -- the same, for the state; ``{count}``
  defaults to 1.
* ``start_research``: ``{tech_folder}`` and the computed click target
  ``research.tech_point``. The tech tree has no search box, so a technology is
  clicked where ``common/technologies`` says it is drawn. See ``tech_point``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..identifiers import IdentifierIndex, TechPlace

#: Click targets computed per call rather than recorded, and the recorded targets
#: each is computed from. ``hoi4-harness calibrate`` asks for the latter.
COMPUTED_TARGETS: dict[str, tuple[str, ...]] = {
    "research.tech_point": ("research.grid.origin", "research.grid.col1", "research.grid.row1"),
}
#: Placeholders the resolver fills, beyond the action's own arguments.
DERIVED_PARAMETERS: dict[str, frozenset[str]] = {
    "set_national_focus": frozenset({"focus_name"}),
    "queue_construction": frozenset({"state_name", "count"}),
    "start_research": frozenset({"tech_folder"}),
}
#: The tech folders 1.19.5 draws a tab for (the ``folder`` names common/technologies
#: uses), so ``research.tab.{tech_folder}`` can be calibrated before it is needed.
TECH_FOLDERS = (
    "infantry_folder", "support_folder", "armour_folder", "artillery_folder",
    "nsb_armour_folder", "air_techs_folder", "bba_air_techs_folder", "naval_folder",
    "industry_folder", "electronics_folder",
)
#: Values a templated click target is calibrated for, per placeholder. Others
#: (an advisor, a production line) have too many values to list and are named by hand.
def template_values() -> dict[str, tuple[str, ...]]:
    from ..actions.catalog import BUILDINGS

    return {"building": tuple(BUILDINGS), "tech_folder": TECH_FOLDERS}


#: The tech tree starts below the topbar and the folder tabs; a point above this
#: row is on the tabs, not on a technology.
MIN_TECH_Y = 150
MIN_FRAGMENT = 3

_TYPABLE = re.compile(r"[A-Za-z0-9'-]+")
_EDGE_PUNCT = ".,:;!?\"()"


class Unresolvable(Exception):
    """The action cannot be performed from the UI as asked; nothing was sent."""


@dataclass
class Resolved:
    arguments: dict
    coordinates: dict[str, tuple[int, int]] = field(default_factory=dict)


def search_fragment(name: str) -> str:
    """The longest run of ASCII words in a localised name, to type into a search box.

    Color codes (``§Y``), scripted loc (``[GetX]``), ``$KEY$`` references and
    escapes are not text the game would match, so they cut the name into runs;
    so do words with characters a US keyboard cannot type. A word that carries
    punctuation at its end ends its run too, since the search would not match
    across the punctuation. Raises Unresolvable when nothing usable is left.
    """
    text = re.sub(r"§.", "", name).replace("\\n", " ")
    best: list[str] = []

    def close(run: list[str]) -> None:
        nonlocal best
        if len(" ".join(run)) > len(" ".join(best)):
            best = run

    for segment in re.split(r"\[[^\]]*\]|\$[^$]*\$", text):
        run: list[str] = []
        for raw in segment.split():
            core = raw.lstrip(_EDGE_PUNCT).rstrip(_EDGE_PUNCT)
            if not _TYPABLE.fullmatch(core):
                close(run)
                run = []
                continue
            run.append(core)
            if raw.rstrip(_EDGE_PUNCT) != raw:      # punctuation after it: the run ends here
                close(run)
                run = []
        close(run)
    fragment = " ".join(best)
    if len(fragment) < MIN_FRAGMENT:
        raise Unresolvable(
            f"{name!r} has no ASCII-only fragment of {MIN_FRAGMENT}+ characters to type into "
            "the search box (accents, scripted names and symbols cannot be typed)"
        )
    return fragment


def tech_point(
    place: TechPlace,
    coordinates: dict[str, tuple[int, int]],
    screen: tuple[int, int] | None,
    tech: str = "",
) -> tuple[int, int]:
    """Where to click a technology: its grid cell, from three calibrated points.

    The tree is a grid of equal cells, columns running across (the year columns:
    1918, 1936, ...) and rows down. Calibrate the centre of cell (0, 0) as
    ``research.grid.origin``, of cell (1, 0) as ``research.grid.col1`` and of cell
    (0, 1) as ``research.grid.row1`` (empty cells are fine: aim, do not click).
    A folder whose cell (0, 0) sits elsewhere takes its own
    ``research.origin.<folder>``, and a technology that does not sit where the
    grid says takes an exact ``research.tech.<id>``. A point off the screen or
    on the tab strip is refused: this driver does not scroll the tree.
    """
    exact = coordinates.get(f"research.tech.{tech}") if tech else None
    if exact is not None:
        return exact
    base = coordinates.get("research.grid.origin")
    col1, row1 = coordinates.get("research.grid.col1"), coordinates.get("research.grid.row1")
    if base is None or col1 is None or row1 is None:
        missing = [t for t in COMPUTED_TARGETS["research.tech_point"] if t not in coordinates]
        raise Unresolvable(
            f"no calibrated coordinate for {', '.join(missing)}. Run: hoi4-harness calibrate "
            + " ".join(missing)
        )
    origin = coordinates.get(f"research.origin.{place.folder}") or base
    step_x, step_y = col1[0] - base[0], row1[1] - base[1]
    x = origin[0] + place.column * step_x
    y = origin[1] + place.row * step_y
    width, height = screen if screen else (1920, 1080)
    if not (0 <= x < width and MIN_TECH_Y <= y < height):
        raise Unresolvable(
            f"{tech or 'the technology'} (column {place.column}, row {place.row} of "
            f"{place.folder}) would be clicked at ({x}, {y}), off the visible tree on a "
            f"{width}x{height} screen; this driver does not scroll the tech tree"
        )
    return int(x), int(y)


def resolve(
    name: str,
    arguments: dict,
    index: IdentifierIndex | None,
    coordinates: dict[str, tuple[int, int]],
    screen: tuple[int, int] | None,
    wanted: set[str] | frozenset[str] = frozenset(),
) -> Resolved:
    """Derive the extra arguments and computed click points for one action.

    ``wanted`` is what the script uses (its placeholders and click targets): only
    those are derived, so a script that never types a focus name does not need
    the localisation to run."""
    args = dict(arguments)
    computed: dict[str, tuple[int, int]] = {}
    if name == "set_national_focus" and "focus_name" in wanted:
        if index is None:
            raise Unresolvable("typing a focus name needs the game's localisation: pass --game-dir")
        focus_id = str(args["focus_id"])
        display = index.focus_name(focus_id)
        if not display:
            raise Unresolvable(f"no English name for focus {focus_id}, so there is nothing to search for")
        args["focus_name"] = search_fragment(display)
    elif name == "queue_construction" and wanted & {"state_name", "count"}:
        if index is None:
            raise Unresolvable("typing a state name needs the game's localisation: pass --game-dir")
        state = index.resolve_state(str(args["state"]))
        if state is None:
            raise Unresolvable(f"no state {args['state']!r} in this playset")
        args["state_name"] = search_fragment(state.name)
        args.setdefault("count", 1)
    elif name == "start_research" and wanted & {"tech_folder", "research.tech_point"}:
        if index is None:
            raise Unresolvable("locating a technology needs common/technologies: pass --game-dir")
        tech = str(args["technology"])
        place = index.tech_places.get(tech)
        if place is None:
            raise Unresolvable(f"no folder position for technology {tech} in this playset")
        args["tech_folder"] = place.folder
        computed["research.tech_point"] = tech_point(place, coordinates, screen, tech)
    return Resolved(args, computed)
