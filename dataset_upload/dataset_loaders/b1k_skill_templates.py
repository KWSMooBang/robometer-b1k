#!/usr/bin/env python3
"""Render BEHAVIOR-1K skill annotations as natural-language subtask instructions.

A skill annotation is a small structured record:

    {"skill_description": ["pick up from"],
     "object_id": [["radio_89", "coffee_table_koagbh_0"]],
     "memory_prefix": ["back"], "spatial_prefix": []}

which this module turns into ``"pick up the radio from the coffee table"`` --
the ``task`` string a Robometer trajectory is trained and queried with.

Three closed vocabularies drive it, all measured over the full 20,000-episode
annotation set:

* 35 ``skill_description`` values, keyed together with object arity because the
  same skill appears with different arities (``pour`` spans 1..11).
* 573 object ids -> 278 display names (see ``b1k_object_names``).
* 3 ``memory_prefix`` and 35 ``spatial_prefix`` values.

``object_id[0]`` is the manipulated object for every skill except ``pour``,
where the manipulated object is the container at index ``-2`` and the contents
occupy ``[0:-2]``. Templates below follow that layout.

Anything not covered returns ``None`` -- callers must drop the segment and count
it rather than fall back to a guess, since a wrong instruction is worse than a
missing one for reward-model training.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional, Sequence

from dataset_upload.dataset_loaders.b1k_object_names import (
    OBJECT_NAME_MAP,
    PSEUDO_OBJECTS,
)

__all__ = [
    "normalize_object",
    "render_instruction",
    "SKILL_ARITIES",
    "MEMORY_PREFIX_MAP",
    "SPATIAL_PREFIX_MAP",
]


# --------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------


def normalize_object(obj_id: str) -> Optional[str]:
    """Display name for an object id, or None if it is not in the vocabulary.

    Unknown ids return None on purpose: the vocabulary is closed, so a miss
    means the dataset changed and ``b1k_object_names`` needs regenerating.
    """
    if not obj_id:
        return None
    return OBJECT_NAME_MAP.get(obj_id)


def _the(name: str) -> str:
    return f"the {name}"


def _join(names: Sequence[str]) -> str:
    """'a', 'b', 'c' -> 'the a, the b and the c' (duplicates collapsed)."""
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    if not seen:
        return ""
    if len(seen) == 1:
        return _the(seen[0])
    return ", ".join(_the(n) for n in seen[:-1]) + f" and {_the(seen[-1])}"


# --------------------------------------------------------------------------
# prefixes
# --------------------------------------------------------------------------

# memory_prefix: 3 values across the whole dataset.
#   "back"      -> adverb attached after the verb phrase's object
#   "the other" -> determiner replacing "the" on the manipulated object
#   "the same"  -> same, 3 occurrences
MEMORY_PREFIX_MAP = {
    "back": "back",
    "the other": "the other",
    "the same": "the same",
}

# spatial_prefix: 35 values. ``None`` means "drop the segment"; "" means
# "no prefix, render normally".
#
#   kind="det"   -> adjective before the target noun:      "the left door"
#   kind="prep"  -> prepositional phrase before target:    "to the edge of the table"
#   kind="part"  -> a named part of the target:            "the right door of the fridge"
#   kind="layer" -> a shelf/level of the target:           "the 4th layer of the shelf"
#   kind="pose"  -> trailing clause on the verb:           "... and face it away"
SPATIAL_PREFIX_MAP: dict[str, Optional[tuple[str, str]]] = {
    "": ("", ""),
    "left": ("det", "left"),
    "right": ("det", "right"),
    "center": ("det", "center"),
    "to_the_edge_of": ("prep", "to the edge of"),
    "in_front_of": ("prep", "in front of"),
    "under": ("prep", "under"),
    "near": ("prep", "near"),
    "away": ("prep", "away from"),
    "left_door": ("part", "left door"),
    "right_door": ("part", "right door"),
    "first_left_door": ("part", "first left door"),
    "first_right_door": ("part", "first right door"),
    "second_left_door": ("part", "second left door"),
    "second_right_door": ("part", "second right door"),
    "right_upper_door": ("part", "upper right door"),
    "right_lower_door": ("part", "lower right door"),
    "high_level": ("layer", "top level"),
    "middle_level": ("layer", "middle level"),
    "low_level": ("layer", "bottom level"),
    "layer_2": ("layer", "2nd layer"),
    "layer_3": ("layer", "3rd layer"),
    "layer_4": ("layer", "4th layer"),
    "layer_5": ("layer", "5th layer"),
    "layer_6": ("layer", "6th layer"),
    "reorient": ("pose", "and reorient it"),
    "face": ("pose", "and face it toward you"),
    "face_away": ("pose", "and face it away"),
}

# Grid coordinates ("2X2", "3X1", "1x3", ...): ~1,800 segments. Rendering a
# shelf-grid cell in natural language is guesswork, so drop those segments
# rather than invent a phrase. Matched by pattern so a new grid size shows up
# as a drop, not as an unknown token.
_GRID_PREFIX = re.compile(r"^\d+[xX]\d+$")


def spatial_kind(token: str) -> Optional[tuple[str, str]]:
    """Resolve a spatial_prefix token.

    Returns the (kind, word) pair, or None when the segment must be dropped.
    Raises KeyError for tokens outside the known vocabulary so that callers can
    tell "deliberately dropped" from "dataset changed under us".
    """
    if token in SPATIAL_PREFIX_MAP:
        return SPATIAL_PREFIX_MAP[token]
    if _GRID_PREFIX.match(token):
        return None
    raise KeyError(token)


def _target_phrase(
    name: str,
    prep: str,
    spatial: Optional[tuple[str, str]],
    *,
    distinct_from: Optional[str] = None,
) -> tuple[str, str]:
    """Render the target noun phrase including its preposition.

    The spatial prefix has to *replace* the template's own preposition rather
    than stack on top of it: ``push to`` + ``to_the_edge_of`` must produce
    "push the toolbox to the edge of the countertop", not "... to to the edge
    of ...". 5,698 segments hit that combination.

    ``distinct_from`` is the manipulated object's name. When the target has the
    same display name (stacking a plate on another plate -- 3,598 segments), the
    determiner becomes "another" so the instruction does not read "place the
    plate on the plate".

    Returns (phrase, trailing_clause).
    """
    lead = f"{prep} " if prep else ""
    det = "another" if distinct_from is not None and distinct_from == name else "the"
    if not spatial or not spatial[0]:
        return f"{lead}{det} {name}", ""
    kind, word = spatial
    if kind == "prep":
        return f"{word} {det} {name}", ""  # replaces `prep`
    if kind == "det":
        return f"{lead}the {word} {name}", ""
    if kind in ("part", "layer"):
        return f"{lead}the {word} of {det} {name}", ""
    if kind == "pose":
        return f"{lead}{det} {name}", f" {word}"
    return f"{lead}{det} {name}", ""


# --------------------------------------------------------------------------
# skill templates
# --------------------------------------------------------------------------
#
# Each entry maps (skill_description, arity) -> callable(objs, ctx) -> str.
# ``objs`` are normalized display names; ``ctx`` carries the resolved prefixes.
#
# Naming convention below: o0 is the manipulated object, o1 the target.


def _t(
    fmt: str,
    *,
    prep: str = "",
    target: int = -1,
    part: str = "",
) -> Callable[[Sequence[str], dict], str]:
    """Build a renderer from a format string.

    Slots:
      ``{o0}..{oN}``  object display names, already carrying "the"
                      (``{o0}`` honours the memory-prefix determiner)
      ``{tgt}``       the spatial target: preposition + noun phrase, with the
                      spatial prefix folded in (see ``_target_phrase``)
      ``{part}``      a named part of the object ("door", "lid", "drawer"),
                      replaced outright by a ``part``-kind spatial prefix so
                      ``open door`` + ``right_door`` reads "open the right door
                      of the fridge"
      ``{back}``      " back" when memory_prefix is "back", else empty

    ``target`` indexes which object fills ``{tgt}``; it is excluded from the
    ``{oN}`` slots only in the sense that templates should not reference it twice.
    """

    def render(objs: Sequence[str], ctx: dict) -> str:
        det = ctx.get("determiner") or "the"
        spatial = ctx.get("spatial")
        fields: dict = {"back": " back" if ctx.get("back") else ""}

        for i, name in enumerate(objs):
            fields[f"o{i}"] = f"{det} {name}" if i == 0 else _the(name)

        trailing = ""
        if "{tgt}" in fmt:
            idx = target if target >= 0 else len(objs) + target
            phrase, trailing = _target_phrase(
                objs[idx],
                prep,
                spatial,
                distinct_from=objs[0] if idx != 0 else None,
            )
            fields["tgt"] = phrase

        if "{part}" in fmt:
            named = part
            if spatial and spatial[0] == "part":
                named = spatial[1]
            elif spatial and spatial[0] in ("det", "layer"):
                named = f"{spatial[1]} {part}"
            fields["part"] = named

        return fmt.format(**fields) + trailing

    return render


def _pour(objs: Sequence[str], ctx: dict) -> str:
    """pour: contents = objs[:-2], container = objs[-2], destination = objs[-1].

    ``manipulating_object_id`` is always ``objs[-2]`` for this skill, which is
    what fixes the layout. The contents list is often long and repetitive
    (4x pepperoni, 2x mushroom halves), so it is elided -- "the contents of the
    tupperware" is shorter and more faithful than enumerating scene instances.
    """
    container, destination = objs[-2], objs[-1]
    target, trailing = _target_phrase(destination, "into", ctx.get("spatial"))
    return f"pour the contents of the {container} {target}{trailing}"


def _sweep_off(objs: Sequence[str], ctx: dict) -> str:
    """sweep off: items = objs[:-1], surface = objs[-1]."""
    surface, trailing = _target_phrase(objs[-1], "off", ctx.get("spatial"))
    return f"sweep {_join(objs[:-1])} {surface}{trailing}"


def _hand_over_hands(objs: Sequence[str], ctx: dict) -> str:
    """hand over with hands: [object, from_hand, to_hand]."""
    return f"hand the {objs[0]} from the {objs[1]} hand to the {objs[2]} hand"


# (skill_description, arity) -> renderer
TEMPLATES: dict[tuple[str, int], Callable[[Sequence[str], dict], str]] = {
    # navigation
    ("move to", 1): _t("go {tgt}", prep="to", target=0),
    # pick and place -- {tgt} is the surface/container the spatial prefix qualifies
    ("pick up from", 2): _t("pick up {o0} {tgt}", prep="from", target=1),
    ("place in", 2): _t("place {o0}{back} {tgt}", prep="in", target=1),
    ("place on", 2): _t("place {o0}{back} {tgt}", prep="on", target=1),
    ("place under", 2): _t("place {o0}{back} {tgt}", prep="under", target=1),
    ("place on next to", 3): _t("place {o0}{back} on {o1} {tgt}", prep="next to", target=2),
    ("place in next to", 3): _t("place {o0}{back} in {o1} {tgt}", prep="next to", target=2),
    ("push to", 2): _t("push {o0} {tgt}", prep="to", target=1),
    ("insert", 2): _t("insert {o0} {tgt}", prep="into", target=1),
    ("attach", 2): _t("attach {o0} {tgt}", prep="to", target=1),
    ("hang", 2): _t("hang {o0} {tgt}", prep="on", target=1),
    ("tip over", 1): _t("tip over {tgt}", target=0),
    ("hold", 1): _t("hold {tgt}", target=0),
    ("release", 1): _t("release {tgt}", target=0),
    ("lift", 1): _t("lift {tgt}", target=0),
    # articulated objects -- {part} absorbs "left_door"/"right_door"/"layer_N"
    ("open door", 1): _t("open the {part} of {o0}", part="door"),
    ("close door", 1): _t("close the {part} of {o0}", part="door"),
    ("open lid", 1): _t("open the {part} of {o0}", part="lid"),
    ("close lid", 1): _t("close the {part} of {o0}", part="lid"),
    ("open drawer", 1): _t("open the {part} of {o0}", part="drawer"),
    ("close drawer", 1): _t("close the {part} of {o0}", part="drawer"),
    ("pull tray", 1): _t("pull out the {part} of {o0}", part="tray"),
    ("push tray", 1): _t("push in the {part} of {o0}", part="tray"),
    ("turn on switch", 1): _t("turn on {tgt}", target=0),
    ("turn off switch", 1): _t("turn off {tgt}", target=0),
    ("press", 1): _t("press {tgt}", target=0),
    # tool use: object_id[0] is the tool, object_id[1] the target
    # (confirmed by manipulating_object_id == object_id[0] for all of these)
    ("chop", 2): _t("chop {tgt} with {o0}", target=1),
    ("spray", 2): _t("spray {tgt} with {o0}", target=1),
    ("wipe hard", 2): _t("wipe {tgt} with {o0}", target=1),
    ("wipe hard", 1): _t("wipe {tgt}", target=0),
    ("ignite", 2): _t("ignite {tgt} with {o0}", target=1),
    ("sweep surface", 2): _t("sweep {tgt} with {o0}", target=1),
    ("sweep surface", 1): _t("sweep {tgt}", target=0),
    ("sweep off", 3): _sweep_off,
    # misc
    ("turn to", 2): _t("turn {o0} {tgt}", prep="toward", target=1),
    ("turn to", 1): _t("turn {tgt} around", target=0),
    ("hand over", 1): _t("hand over {tgt}", target=0),
    ("hand over", 3): _hand_over_hands,
    ("pour", 3): _pour,
    ("pour", 4): _pour,
    ("pour", 6): _pour,
    ("pour", 8): _pour,
    ("pour", 11): _pour,
}

# Arities deliberately left uncovered (segment is dropped and counted):
#   ("pour", 1)             200 segments -- a single object, no container to pour from
#   ("place on next to", 2)  65 segments -- missing the "next to" reference
SKILL_ARITIES = sorted({skill for skill, _ in TEMPLATES})


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def _flatten(value) -> list:
    out: list = []
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten(item))
    elif value is not None:
        out.append(value)
    return out


def render_instruction(
    skill_description: str,
    object_ids: Iterable,
    memory_prefix: Optional[Iterable] = None,
    spatial_prefix: Optional[Iterable] = None,
    task_name: Optional[str] = None,
    include_task_context: bool = False,
) -> Optional[str]:
    """Render one skill annotation as a subtask instruction.

    Returns None when the segment cannot be rendered faithfully -- unknown
    skill, unsupported arity, unknown object, or a dropped spatial prefix.
    Callers must drop and count those segments.
    """
    if not skill_description:
        return None

    raw_objects = _flatten(list(object_ids or []))
    if not raw_objects:
        return None

    names: list[str] = []
    for obj in raw_objects:
        if obj in PSEUDO_OBJECTS:
            names.append(obj)
            continue
        name = normalize_object(obj)
        if name is None:
            return None
        names.append(name)

    renderer = TEMPLATES.get((skill_description, len(names)))
    if renderer is None:
        return None

    memory = _flatten(list(memory_prefix or []))
    spatial = _flatten(list(spatial_prefix or []))

    ctx: dict = {"determiner": "the", "back": False, "spatial": None}
    for token in memory:
        mapped = MEMORY_PREFIX_MAP.get(token)
        if mapped is None:
            return None
        if mapped == "back":
            ctx["back"] = True
        else:
            ctx["determiner"] = mapped

    for token in spatial:
        try:
            mapped = spatial_kind(token)
        except KeyError:
            return None  # vocabulary changed -- fail closed
        if mapped is None:  # grid coordinate -> drop
            return None
        if mapped[0]:
            ctx["spatial"] = mapped

    try:
        instruction = renderer(names, ctx)
    except (IndexError, KeyError):
        return None

    instruction = " ".join(instruction.split())
    if not instruction:
        return None

    if include_task_context and task_name:
        instruction = f"{instruction}. (task: {task_name.replace('_', ' ')})"
    return instruction
