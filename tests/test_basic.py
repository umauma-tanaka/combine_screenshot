import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stitcher import Stitcher


def test_stitch_basic(tmp_path: Path) -> None:
    out = Stitcher().stitch("tests/assets", output=tmp_path / "stitched.png")
    assert out.exists() and out.suffix == ".png"
