import pytest

from cvdc.data.filenames import discover_exercise_dirs, parse_name
from cvdc.data.labels import quality_class


@pytest.mark.parametrize("name, expect", [
    ("E01_P01_AF_VFL_GM.mp4", ("E01", "P01", "F", "FL", "M")),
    ("E01 P01 AF VFL GM", ("E01", "P01", "F", "FL", "M")),           # space separated (paper notation)
    ("E01_P26_Al_VFL_GF", ("E01", "P26", "L", "FL", "F")),           # CSV typo: lowercase l
    ("e02_p7_ar_vo_gf.MP4", ("E02", "P07", "R", "O", "F")),           # case-insensitive, P7 -> P07
    ("E09_P60_AR_VLR_GF", ("E09", "P60", "R", "LR", "F")),
])
def test_parse_valid(name, expect):
    d = parse_name(name)
    assert (d["exercise"], d["subject"], d["view"], d["variation"], d["gender"]) == expect
    assert d["video_id"] == f"{expect[0]}_{expect[1]}_A{expect[2]}_V{expect[3]}_G{expect[4]}"


def test_parse_unknown_variation_is_kept():
    d = parse_name("E03_P41_AF_VP_GM")
    assert d["variation"] == "unknown" and d["variation_raw"] == "P" and d["video_id"] == "E03_P41_AF_VP_GM"


@pytest.mark.parametrize("name", ["E01_P01_AX_VFL_GM", "E1_P01_AF_VFL_GM", "E01_P01_AF_VFL", "Abduction .mp4"])
def test_parse_invalid(name):
    assert parse_name(name) is None


def test_discover_exercise_dirs(tmp_path):
    for d in ["E01(Abduction)", "E02(Aduction)", "__E03(Lateral_Rotation)", "Reference-Model Videos by PTs", "E05x"]:
        (tmp_path / d).mkdir()
    assert sorted(discover_exercise_dirs(tmp_path)) == ["E01", "E02"]
    assert sorted(discover_exercise_dirs(tmp_path, include=["E02", "E07"])) == ["E02"]  # missing E07: no error


def test_quality_class_bins():
    out = quality_class([0, 49.99, 50, 79.99, 80, 100, float("nan")], [50, 80],
                        ["incorrect", "partially_correct", "correct"])
    assert list(out) == ["incorrect", "incorrect", "partially_correct", "partially_correct", "correct", "correct", None]
