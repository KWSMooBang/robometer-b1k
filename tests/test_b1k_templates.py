"""Tests for the BEHAVIOR-1K subtask instruction templates."""

import pytest

from dataset_upload.dataset_loaders.b1k_skill_templates import (
    normalize_object,
    render_instruction,
)


# --------------------------------------------------------------------------
# object normalization
# --------------------------------------------------------------------------


def test_normalize_hashed_and_numbered_ids():
    assert normalize_object("radio_89") == "radio"
    assert normalize_object("coffee_table_koagbh_0") == "coffee table"
    assert normalize_object("broom_172") == "broom"
    assert normalize_object("floors") == "floors"


def test_six_letter_category_suffix_is_not_mistaken_for_a_hash():
    # A regex that strips any trailing 6-letter token turns these into
    # "allen", "bell", "swiss", "wicker", "digital". The audited lookup
    # table is what prevents it.
    assert normalize_object("allen_wrench_189") == "allen wrench"
    assert normalize_object("bell_pepper_213") == "bell pepper"
    assert normalize_object("swiss_cheese_77") == "swiss cheese"
    assert normalize_object("wicker_basket_178") == "wicker basket"
    assert normalize_object("digital_camera_79") == "digital camera"
    assert normalize_object("chocolate_chip_cookie_207") == "chocolate chip cookie"


def test_normalize_bare_categories_with_six_letter_tails():
    assert normalize_object("wine_bottle") == "wine bottle"
    assert normalize_object("toy_figure") == "toy figure"


def test_normalize_repeated_numeric_suffixes():
    assert normalize_object("half_beet_211_0") == "half beet"
    assert normalize_object("half-log-176-0") == "half log"


def test_normalize_double_underscore():
    assert normalize_object("diced__vidalia_onion") == "diced vidalia onion"


def test_unknown_id_returns_none():
    assert normalize_object("totally_unseen_thing_zzz") is None
    assert normalize_object("") is None


# --------------------------------------------------------------------------
# instruction rendering
# --------------------------------------------------------------------------


def test_render_pick_up_from():
    assert (
        render_instruction("pick up from", [["radio_89", "coffee_table_koagbh_0"]], [], [])
        == "pick up the radio from the coffee table"
    )


def test_render_move_to():
    assert render_instruction("move to", [["radio_89"]], [], []) == "go to the radio"


def test_memory_prefix_back():
    assert (
        render_instruction("place on", [["radio_89", "coffee_table_koagbh_0"]], ["back"], [])
        == "place the radio back on the coffee table"
    )


def test_memory_prefix_the_other():
    assert (
        render_instruction("pick up from", [["radio_89", "coffee_table_koagbh_0"]], ["the other"], [])
        == "pick up the other radio from the coffee table"
    )


def test_tool_skill_puts_target_first():
    # object_id[0] is the tool (manipulating_object_id confirms this)
    assert (
        render_instruction("chop", [["carving_knife_209", "head_cabbage_212"]], [], [])
        == "chop the head cabbage with the carving knife"
    )
    assert (
        render_instruction("sweep surface", [["broom_172", "floors_ghetev_0"]], [], [])
        == "sweep the floors with the broom"
    )


def test_pour_elides_contents():
    objs = [["pepperoni_85", "pepperoni_86", "tupperware_73", "bowl_76"]]
    assert (
        render_instruction("pour", objs, [], [])
        == "pour the contents of the tupperware into the bowl"
    )


def test_sweep_off_collapses_duplicate_items():
    objs = [["half_log_176_0", "half_log_176_1", "driveway_umalys_0"]]
    assert render_instruction("sweep off", objs, [], []) == "sweep the half log off the driveway"


def test_hand_over_with_hands():
    objs = [["hinged_jar_236", "left", "right"]]
    assert (
        render_instruction("hand over", objs, [], [])
        == "hand the hinged jar from the left hand to the right hand"
    )


def test_spatial_prefix_part_replaces_the_generic_part():
    # ~6,000 segments. Must not read "open the door of the right door of ...".
    assert (
        render_instruction("open door", [["fridge_petcxr_0"]], [], ["right_door"])
        == "open the right door of the fridge"
    )
    assert (
        render_instruction("close door", [["fridge_petcxr_0"]], [], ["left_door"])
        == "close the left door of the fridge"
    )


def test_spatial_prefix_prep_replaces_the_template_preposition():
    # 5,698 segments. Must not read "push the toolbox to to the edge of ...".
    assert (
        render_instruction("push to", [["toolbox_191", "countertop_fjkase_0"]], [], ["to_the_edge_of"])
        == "push the toolbox to the edge of the countertop"
    )
    assert (
        render_instruction(
            "place on next to",
            [["cauldron_92", "floors_ulujpr_0", "coffee_table_koagbh_0"]],
            [],
            ["in_front_of"],
        )
        == "place the cauldron on the floors in front of the coffee table"
    )


def test_spatial_prefix_layer():
    assert (
        render_instruction("place in", [["plate_207", "top_cabinet_lkxmne_1"]], [], ["layer_2"])
        == "place the plate in the 2nd layer of the top cabinet"
    )
    assert (
        render_instruction("pick up from", [["plate_207", "shelf_fjozhc_0"]], [], ["high_level"])
        == "pick up the plate from the top level of the shelf"
    )


def test_spatial_prefix_determiner():
    assert (
        render_instruction("pick up from", [["plate_207", "countertop_fjkase_0"]], [], ["left"])
        == "pick up the plate from the left countertop"
    )


def test_spatial_prefix_empty_string_is_ignored():
    assert render_instruction("move to", [["radio_89"]], [], [""]) == "go to the radio"


# --------------------------------------------------------------------------
# failure modes -- must return None, never a guess
# --------------------------------------------------------------------------


def test_arity_mismatch_returns_none():
    assert render_instruction("pick up from", [["radio_89"]], [], []) is None


def test_unknown_skill_returns_none():
    assert render_instruction("teleport", [["radio_89"]], [], []) is None


def test_unknown_object_returns_none():
    assert render_instruction("move to", [["not_a_real_object"]], [], []) is None


def test_grid_spatial_prefix_is_dropped():
    assert render_instruction("place on", [["radio_89", "coffee_table_koagbh_0"]], [], ["2X2"]) is None


def test_empty_inputs_return_none():
    assert render_instruction("", [["radio_89"]], [], []) is None
    assert render_instruction("move to", [], [], []) is None


def test_nested_object_lists_are_flattened():
    assert (
        render_instruction("pick up from", [[["radio_89"], ["coffee_table_koagbh_0"]]], [], [])
        == "pick up the radio from the coffee table"
    )


@pytest.mark.parametrize(
    "skill,objs",
    [
        ("move to", [["radio_89"]]),
        ("place in", [["can_of_soda_114", "trash_can_116"]]),
        ("open lid", [["hinged_jar_235"]]),
        ("turn on switch", [["lighter_73"]]),
        ("insert", [["pen_229", "pencil_case_224"]]),
    ],
)
def test_common_skills_render(skill, objs):
    out = render_instruction(skill, objs, [], [])
    assert out and out == out.strip() and "  " not in out


def test_same_category_target_uses_another():
    # "place the plate on the plate" (3,598 segments) is ambiguous; stacking
    # onto a second instance should read "another".
    assert (
        render_instruction("place on", [["plate_207", "plate_208"]], [], [])
        == "place the plate on another plate"
    )


def test_layer_6_is_mapped():
    assert (
        render_instruction("place in", [["plate_207", "shelf_fjozhc_0"]], [], ["layer_6"])
        == "place the plate in the 6th layer of the shelf"
    )


def test_unseen_grid_size_is_dropped_not_unknown():
    from dataset_upload.dataset_loaders.b1k_skill_templates import spatial_kind

    assert spatial_kind("9X9") is None
    assert spatial_kind("1x3") is None
    with pytest.raises(KeyError):
        spatial_kind("some_new_token")
