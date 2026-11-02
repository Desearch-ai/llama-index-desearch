#!/usr/bin/env python3
"""Live smoke test for youtube_tool, reddit_tool and threads_tool (llama-index-desearch 0.3.0).

Calls every action once against the real Desearch API, reads one more page
wherever a list action returns a next_cursor, and prints one line per call:
``PASS name`` or ``FAIL name: error``. Exits with status 1 if any call fails,
2 if the API key is not set.

Usage::

    DESEARCH_API_KEY=... python scripts/e2e-youtube-reddit-threads.py
    python scripts/e2e-youtube-reddit-threads.py --base-url https://api-dev.desearch.ai \\
        --key-env DESEARCH_DEV_API_KEY

The key is read from the variable named by ``--key-env`` (default
``DESEARCH_API_KEY``), or from ``~/.config/desearch/e2e.env``. Each call is
billed as one request ($0.00207); a run is about 23 requests. An HTTP 5xx is
retried once, because the upstream provider sometimes returns a transient 503.

The script imports the package from this working tree, not from site-packages.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index_desearch.tools import DesearchToolSpec  # noqa: E402

# Public inputs that exist. The post IDs are fallbacks, used only when the
# search before them returns nothing to take an ID from.
QUERY = "bittensor"
YOUTUBE_VIDEO_ID = "dQw4w9WgXcQ"
YOUTUBE_CHANNEL = "@MrBeast"
SUBREDDIT = "AskReddit"
FALLBACK_REDDIT_POST_ID = "1q6pxwn"
THREADS_USER = "zuck"
FALLBACK_THREADS_CODE = "DK-BydcJHkF"


def _load_env_file() -> None:
    path = Path.home() / ".config" / "desearch" / "e2e.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value and not os.environ.get(key):
            os.environ[key] = value


class SmokeRun:
    def __init__(self, spec: DesearchToolSpec) -> None:
        self.spec = spec
        self.failures = 0

    def report(self, name: str, error: Optional[str] = None, detail: str = "") -> None:
        if error is None:
            print(f"PASS {name}" + (f" ({detail})" if detail else ""), flush=True)
        else:
            self.failures += 1
            print(f"FAIL {name}: {error}", flush=True)

    def call(
        self,
        name: str,
        make_call: Callable[[], Any],
        check: Callable[[Any], str] = lambda result: "",
    ) -> Any:
        """Run one call. A string result is the function's error text; a 5xx is retried once."""
        try:
            result = make_call()
            if isinstance(result, str) and result.startswith("HTTP 5"):
                print(f"RETRY {name}: {result[:120]}", flush=True)
                result = make_call()
            if isinstance(result, str):
                self.report(name, result)
                return None
            detail = check(result)
        except Exception as exc:  # noqa: BLE001 - report every failure and go on
            self.report(name, f"{type(exc).__name__}: {exc}")
            return None
        self.report(name, detail=detail)
        return result

    def social(
        self, tool: str, action: str, paginated: bool, label: str = "", **arguments: Any
    ) -> Any:
        """Run one action; for a list with a next_cursor, read one more page."""
        function = getattr(self.spec, tool)
        name = f"{tool} {action}" + (f" ({label})" if label else "")

        def check(result: Any) -> str:
            if not isinstance(result, dict):
                raise TypeError(f"expected a dict, got {type(result).__name__}")
            if "results" not in result:
                return ", ".join(sorted(result)[:4])
            if not isinstance(result["results"], list):
                raise TypeError(f"expected a list of results, got {result['results']!r:.200}")
            cursor = result.get("next_cursor")
            if cursor and not paginated:
                raise AssertionError("this action should not return a cursor")
            return f"{len(result['results'])} items, next_cursor={'yes' if cursor else 'no'}"

        page = self.call(name, lambda: function(action=action, **arguments), check)
        if not isinstance(page, dict) or "results" not in page:
            return page
        items: List[Any] = list(page["results"])
        cursor = page.get("next_cursor")
        if paginated and cursor:
            next_page = self.call(
                f"{name} [page 2]",
                lambda: function(action=action, cursor=cursor, **arguments),
                check,
            )
            if next_page:
                items.extend(next_page["results"])
        return items


def _first(items: Any, key: str, fallback: str) -> str:
    for item in items or []:
        if isinstance(item, dict) and item.get(key):
            return str(item[key])
    return fallback


def run_youtube(run: SmokeRun) -> None:
    run.social("youtube_tool", "search", True, query=QUERY, type="videos")
    run.social("youtube_tool", "video", False, id=YOUTUBE_VIDEO_ID)
    run.social(
        "youtube_tool", "video", False, "URL id", id=f"https://youtu.be/{YOUTUBE_VIDEO_ID}"
    )
    run.social("youtube_tool", "transcript", False, id=YOUTUBE_VIDEO_ID, language="en")
    run.social("youtube_tool", "comments", True, id=YOUTUBE_VIDEO_ID, order="top")
    run.social("youtube_tool", "channel", False, handle=YOUTUBE_CHANNEL)
    run.social("youtube_tool", "channel_videos", True, handle=YOUTUBE_CHANNEL, sort="latest")


def run_reddit(run: SmokeRun) -> None:
    posts = run.social("reddit_tool", "search", True, query=QUERY)
    run.social("reddit_tool", "subreddit", False, name=SUBREDDIT)
    run.social(
        "reddit_tool", "subreddit_posts", True, name=f"r/{SUBREDDIT}", sort="top", timeframe="week"
    )
    post_id = _first(posts, "id", FALLBACK_REDDIT_POST_ID)
    run.social("reddit_tool", "post", False, id=post_id)
    run.social("reddit_tool", "comments", True, id=post_id)


def run_threads(run: SmokeRun) -> None:
    search = run.social("threads_tool", "search", False, query=QUERY)
    run.social("threads_tool", "search_users", False, query=THREADS_USER)
    run.social("threads_tool", "profile", False, username=f"@{THREADS_USER}")
    posts = run.social("threads_tool", "profile_posts", False, username=THREADS_USER)
    code = _first(posts, "code", "") or _first(search, "code", FALLBACK_THREADS_CODE)
    run.social("threads_tool", "post", False, id=code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DESEARCH_BASE_URL", "https://api.desearch.ai"),
        help="API base URL (default: $DESEARCH_BASE_URL or production)",
    )
    parser.add_argument(
        "--key-env",
        default="DESEARCH_API_KEY",
        help="environment variable that holds the API key (default: DESEARCH_API_KEY)",
    )
    args = parser.parse_args()

    _load_env_file()
    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        print(f"{args.key_env} is not set", file=sys.stderr)
        return 2

    spec = DesearchToolSpec(api_key=api_key)
    spec.client.base_url = args.base_url.rstrip("/")
    run = SmokeRun(spec)
    run_youtube(run)
    run_reddit(run)
    run_threads(run)

    print(f"{run.failures} failure(s) on {args.base_url}", flush=True)
    return 1 if run.failures else 0


if __name__ == "__main__":
    sys.exit(main())
