#!/usr/bin/env python3
"""Live smoke test of every DesearchToolSpec function.

Calls the real Desearch API once per function (and once per TikTok, Facebook
and Instagram action) with the smallest counts, reads one more page wherever a
list comes back with a next_cursor, and prints one line per call:
``PASS name`` or ``FAIL name: error``. Exits 1 if any call fails, 2 if
``DESEARCH_API_KEY`` is not set.

Usage::

    DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py
    DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py --base-url https://api.desearch.ai

These calls are billed at normal prices (a few cents in total). People Search
is a paid background job and is never started unless you pass
``--with-people``; that option starts one TikTok Creator Search with the
minimum count (5) and cancels it at once. ``people_profiles_tool`` is free and
always runs.

The script imports the package from this working tree, not from site-packages.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desearch_py import Desearch, PeopleTaskConflictError  # noqa: E402

from llama_index_desearch.tools import DesearchToolSpec  # noqa: E402

# Public inputs that exist. The IDs are fallbacks, used only when the call
# before them returns nothing to take an ID from.
X_USER = "elonmusk"
TIKTOK_USER = "tiktok"
INSTAGRAM_USER = "instagram"
FACEBOOK_PAGE = "facebook"
HASHTAG = "ai"
FALLBACK_X_POST_ID = "1892527552029499853"
FALLBACK_X_POST_URL = "https://x.com/RacingTriple/status/1892527552029499853"
FALLBACK_TIKTOK_VIDEO_ID = "7620891413932166433"
FALLBACK_INSTAGRAM_SHORTCODE = "DY79rTVS6OK"
FALLBACK_FACEBOOK_POST_ID = "1525856912919657"


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
        """Run one call. A string result is the function's error text."""
        try:
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

    def social(self, tool: str, action: str, paginated: bool, **arguments: Any) -> List[Any]:
        """Run one social action; for a list with a next_cursor, read one more page."""
        function = getattr(self.spec, tool)
        name = f"{tool} {action}"

        def check(result: Any) -> str:
            if not isinstance(result, dict):
                raise TypeError(f"expected a dict, got {type(result).__name__}")
            if "results" not in result:
                return ""
            if not isinstance(result["results"], list):
                raise TypeError(f"expected a list of results, got {result['results']!r:.200}")
            cursor = result.get("next_cursor")
            if cursor and not paginated:
                raise AssertionError("this action should not return a cursor")
            return f"{len(result['results'])} items, next_cursor={'yes' if cursor else 'no'}"

        page = self.call(name, lambda: function(action=action, **arguments), check)
        if not isinstance(page, dict) or "results" not in page:
            return [page] if page else []
        items = list(page["results"])
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


def _count(key: str) -> Callable[[Any], str]:
    def check(result: Any) -> str:
        items = result.get(key) if isinstance(result, dict) else result
        if not isinstance(items, list):
            raise TypeError(f"expected a list, got {type(items).__name__}")
        return f"{len(items)} items"

    return check


def run_search(run: SmokeRun) -> None:
    s = run.spec
    run.call(
        "ai_search_tool",
        lambda: s.ai_search_tool(prompt="What is Bittensor?", tool=["web"]),
        lambda r: "summary" if r.get("completion") or r.get("text") else "",
    )
    run.call(
        "ai_web_links_search_tool",
        lambda: s.ai_web_links_search_tool(prompt="Bittensor subnets", count=10),
    )
    run.call(
        "ai_x_links_search_tool",
        lambda: s.ai_x_links_search_tool(prompt="Bittensor", count=10),
    )
    run.call(
        "search_api_tool",
        lambda: s.search_api_tool(query="What is Bittensor?", mode="fast", count=10),
        _count("results"),
    )
    run.call(
        "web_search_tool", lambda: s.web_search_tool(query="Bittensor", num=1), _count("")
    )
    run.call(
        "extract_tool",
        lambda: s.extract_tool(url="https://example.com"),
        lambda r: f"{len(r['content'])} characters",
    )


def run_x(run: SmokeRun) -> None:
    s = run.spec
    found = run.call(
        "twitter_search_tool",
        lambda: s.twitter_search_tool(query="bittensor", count=1),
        _count(""),
    )
    timeline = run.call(
        "x_user_posts_tool", lambda: s.x_user_posts_tool(username=X_USER), _count("tweets")
    )
    if isinstance(timeline, dict) and timeline.get("next_cursor"):
        run.call(
            "x_user_posts_tool [page 2]",
            lambda: s.x_user_posts_tool(username=X_USER, cursor=timeline["next_cursor"]),
            _count("tweets"),
        )
    tweets = (timeline or {}).get("tweets") or found or []
    post_id = _first(tweets, "id", FALLBACK_X_POST_ID)
    post_url = _first(tweets, "url", FALLBACK_X_POST_URL)
    run.call("x_post_by_id_tool", lambda: s.x_post_by_id_tool(id=post_id))
    run.call(
        "x_posts_by_urls_tool", lambda: s.x_posts_by_urls_tool(urls=[post_url]), _count("")
    )
    run.call(
        "x_posts_by_user_tool",
        lambda: s.x_posts_by_user_tool(user=X_USER, count=1),
        _count(""),
    )
    run.call(
        "x_post_retweeters_tool",
        lambda: s.x_post_retweeters_tool(id=post_id),
        _count("users"),
    )
    run.call(
        "x_user_replies_tool",
        lambda: s.x_user_replies_tool(user=X_USER, count=1),
        _count(""),
    )
    run.call(
        "x_post_replies_tool",
        lambda: s.x_post_replies_tool(post_id=post_id, count=1),
        _count(""),
    )
    run.call(
        "x_trends_tool", lambda: s.x_trends_tool(woeid=1, count=30), _count("trends")
    )


def run_tiktok(run: SmokeRun) -> None:
    run.social("tiktok_tool", "search", True, query=HASHTAG, count=2)
    run.social("tiktok_tool", "profile", False, username=TIKTOK_USER)
    posts = run.social("tiktok_tool", "profile_posts", True, username=TIKTOK_USER, count=2)
    video_id = _first(posts, "id", FALLBACK_TIKTOK_VIDEO_ID)
    run.social("tiktok_tool", "post", False, id=video_id)
    run.social("tiktok_tool", "comments", True, id=video_id, count=2)
    run.social("tiktok_tool", "hashtag", True, tag=HASHTAG, count=2)
    run.social("tiktok_tool", "trending", False, region="US", count=2)


def run_instagram(run: SmokeRun) -> None:
    run.social("instagram_tool", "search", False, query=INSTAGRAM_USER, count=2)
    run.social("instagram_tool", "profile", False, username=INSTAGRAM_USER)
    posts = run.social(
        "instagram_tool", "profile_posts", True, username=INSTAGRAM_USER, count=2
    )
    shortcode = _first(posts, "shortcode", FALLBACK_INSTAGRAM_SHORTCODE)
    run.social("instagram_tool", "post", False, id=shortcode)
    run.social("instagram_tool", "comments", True, id=shortcode, count=2)
    run.social("instagram_tool", "hashtag", True, tag=HASHTAG, count=2)


def run_facebook(run: SmokeRun) -> None:
    run.social("facebook_tool", "search", True, query="running", count=2)
    run.social("facebook_tool", "profile", False, profile_id=FACEBOOK_PAGE)
    posts = run.social(
        "facebook_tool", "profile_posts", True, profile_id=FACEBOOK_PAGE, count=2
    )
    post_id = _first(posts, "id", FALLBACK_FACEBOOK_POST_ID)
    run.social("facebook_tool", "post", False, id=post_id)
    run.social("facebook_tool", "comments", True, id=post_id, count=2)
    run.social("facebook_tool", "hashtag", True, tag=HASHTAG, count=2)


def run_people(run: SmokeRun, with_people: bool, api_key: str, base_url: str) -> None:
    s = run.spec
    run.call(
        "people_profiles_tool",
        lambda: s.people_profiles_tool(platform="tiktok", limit=1),
        _count("people"),
    )
    if not with_people:
        return

    # Opt-in: start one paid search without waiting, then cancel it.
    started: Dict[str, Any] = run.call(
        "people_search_tool",
        lambda: s.people_search_tool(
            query="Creators who review AI search tools",
            platform="tiktok",
            count=5,
            timeout=0,
        ),
        lambda r: f"task {r['task_id']} {r.get('status')}",
    )
    if not started:
        return

    async def cancel() -> Any:
        async with Desearch(api_key=api_key, base_url=base_url) as client:
            try:
                return await client.cancel_people_task(started["task_id"])
            except PeopleTaskConflictError as exc:  # it already ended
                return exc.task

    run.call(
        "cancel People Search (desearch-py)",
        lambda: asyncio.run(cancel()),
        lambda task: f"status {getattr(task, 'status', 'ended')}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DESEARCH_BASE_URL", "https://api.desearch.ai"),
        help="API base URL (default: $DESEARCH_BASE_URL or production)",
    )
    parser.add_argument(
        "--with-people",
        action="store_true",
        help="also start one paid People Search (count=5) and cancel it at once",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DESEARCH_API_KEY", "").strip()
    if not api_key:
        print("DESEARCH_API_KEY is not set", file=sys.stderr)
        return 2

    spec = DesearchToolSpec(api_key=api_key)
    spec.client.base_url = args.base_url.rstrip("/")
    run = SmokeRun(spec)
    run_search(run)
    run_x(run)
    run_tiktok(run)
    run_instagram(run)
    run_facebook(run)
    run_people(run, args.with_people, api_key, spec.client.base_url)

    print(f"{run.failures} failure(s)", flush=True)
    return 1 if run.failures else 0


if __name__ == "__main__":
    sys.exit(main())
