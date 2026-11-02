"""Package metadata kept in step across pyproject.toml, setup.py and the CHANGELOG."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.3.0"


def test_version_is_the_same_everywhere():
    pyproject = (ROOT / "pyproject.toml").read_text()
    setup_py = (ROOT / "setup.py").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert re.findall(r'^version = "([^"]+)"', pyproject, re.M) == [VERSION]
    assert re.findall(r'version="([^"]+)"', setup_py) == [VERSION]
    assert re.search(r"^## \[([^\]]+)\]", changelog, re.M).group(1) == VERSION


def test_dependencies_need_desearch_py_1_11():
    for name in ("pyproject.toml", "setup.py"):
        text = (ROOT / name).read_text()
        # The YouTube, Reddit and Threads methods arrive in desearch-py 1.11.0
        # (after the TikTok, Facebook, Instagram and Search API methods of 1.8.0).
        assert '"desearch-py>=1.11.0,<2.0.0"' in text, name
        # People Search runs on desearch-py too; nothing calls the API with requests.
        assert "requests" not in text, name
        # desearch-py 1.8.0 and later need Python 3.10.
        assert ">=3.10" in text, name
