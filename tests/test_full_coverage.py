"""Every spec function against a mocked desearch-py client (no HTTP at all).

``FakeDesearch`` replaces ``desearch_py.Desearch`` inside ``llama_index_desearch.tools``.
It only answers methods the real client has, records each call with its
arguments, and returns the kind of object the real method returns.
"""

import ast
import asyncio
from pathlib import Path

import pytest
from desearch_py import Desearch as RealDesearch
from desearch_py import (
    DesearchAPIError,
    DesearchCostMetadata,
    DesearchResponse,
    FacebookComment,
    FacebookPost,
    FacebookProfile,
    InstagramComment,
    InstagramMedia,
    InstagramProfile,
    InstagramSearchResult,
    PeopleProfilesPage,
    PeopleResultsPage,
    PeopleTask,
    SearchApiResponse,
    TikTokComment,
    TikTokPost,
    TikTokProfile,
    TwitterScraperTweet,
    WebSearchResponse,
    WebSearchResultsResponse,
    XLinksSearchResponse,
    XRetweetersResponse,
    XTrendsResponse,
    XUserPostsResponse,
)

from llama_index_desearch import tools as tools_module
from llama_index_desearch.tools import _SOCIAL_ACTIONS, DesearchToolSpec

PACKAGE = Path(tools_module.__file__).resolve().parent
LEGACY_METHODS = {"search", "basic_twitter_search", "basic_web_search"}

TWEET = TwitterScraperTweet(
    id="1850000000000000000",
    text="Subnet 22 is live.",
    reply_count=1,
    retweet_count=2,
    like_count=3,
    quote_count=0,
    bookmark_count=0,
    created_at="2026-10-01T12:00:00Z",
    user={"id": "42", "username": "desearch_ai"},
)
TWEET_DICT = TWEET.model_dump(mode="json", exclude_none=True)
TASK = PeopleTask(
    task_id="5b33f3ee-168b-4780-9c8e-63a626781872",
    search_type="x_profile_search",
    platform="x",
    status="completed",
    query="Indie hackers building in public",
    count=25,
    result_count=0,
    reserved_cost_usd=1.0,
    created_at="2026-10-19T10:00:00Z",
)


def _page(items, cursor=None):
    return DesearchResponse(
        data=items, metadata=DesearchCostMetadata(cost_usd=0.002, next_cursor=cursor)
    )


# What each desearch-py method returns. Social list methods return a page with
# its metadata, because the spec always asks for include_metadata=True.
RESPONSES = {
    "ai_search": {"text": "Bittensor is a network.", "completion": "Summary."},
    "ai_web_links_search": WebSearchResponse(
        search_results=[
            {"title": "Bittensor", "link": "https://bittensor.com", "snippet": "AI network"}
        ]
    ),
    "ai_x_links_search": XLinksSearchResponse(miner_tweets=[TWEET]),
    "search_api": SearchApiResponse(
        results=[{"url": "https://bittensor.com", "title": "Bittensor"}], cost_usd=0.001
    ),
    "web_search": WebSearchResultsResponse(
        data=[{"title": "One", "link": "https://one.example", "snippet": "first"}]
    ),
    "extract": "Page text.",
    "x_search": [TWEET],
    "x_post_by_id": TWEET,
    "x_posts_by_urls": [TWEET],
    "x_posts_by_user": [TWEET],
    "x_post_retweeters": XRetweetersResponse(
        users=[{"id": "7", "username": "fan"}], next_cursor="rt-2"
    ),
    "x_user_posts": XUserPostsResponse(
        user={"id": "42", "username": "desearch_ai"}, tweets=[TWEET], next_cursor="tl-2"
    ),
    "x_user_replies": [TWEET],
    "x_post_replies": [TWEET],
    "x_trends": XTrendsResponse(trends=[{"name": "#AI", "rank": 1}]),
    "tiktok_search": _page([TikTokPost(id="v1", text="a")], "tt-2"),
    "tiktok_post_by_id": TikTokPost(id="v1", text="a"),
    "tiktok_profile": TikTokProfile(id="u1", username="tiktok"),
    "tiktok_profile_posts": _page([TikTokPost(id="v1")], "tt-2"),
    "tiktok_comments": _page([TikTokComment(id="c1", text="nice")], "tt-2"),
    "tiktok_hashtag": _page([TikTokPost(id="v1")], "tt-2"),
    "tiktok_trending": _page([TikTokPost(id="v9")]),
    "facebook_search": _page([FacebookPost(id="ad1")], "fb-2"),
    "facebook_post_by_id": FacebookPost(id="p1", text="hello"),
    "facebook_profile": FacebookProfile(id="20531316728", username="facebook"),
    "facebook_profile_posts": _page([FacebookPost(id="p1")], "fb-2"),
    "facebook_comments": _page([FacebookComment(id="c1")], "fb-2"),
    "facebook_hashtag": _page([FacebookPost(id="ad2")], "fb-2"),
    "instagram_search": _page([InstagramSearchResult(type="user", username="instagram")]),
    "instagram_media": InstagramMedia(id="m1", shortcode="DY79rTVS6OK"),
    "instagram_profile": InstagramProfile(id="25025320", username="instagram"),
    "instagram_profile_posts": _page([InstagramMedia(id="m1")], "ig-2"),
    "instagram_comments": _page([InstagramComment(id="c1")], "ig-2"),
    "instagram_hashtag": _page([InstagramMedia(id="m2")], "ig-2"),
    "x_profile_search": TASK,
    "youtube_creator_search": TASK,
    "linkedin_profile_search": TASK,
    "tiktok_creator_search": TASK,
    "get_people_task": TASK,
    "get_people_results": PeopleResultsPage(
        task_id=TASK.task_id, status="completed", results=[], next_cursor="r1", has_more=False
    ),
    "list_people_profiles": PeopleProfilesPage(profiles=[], total=0),
}


class Recorder:
    def __init__(self):
        self.calls = []
        self.clients = []
        self.responses = dict(RESPONSES)

    def methods(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()

    class FakeDesearch:
        def __init__(self, api_key, base_url="https://api.desearch.ai"):
            self.api_key = api_key
            self.base_url = base_url
            self.closed = False
            rec.clients.append(self)

        async def close(self):
            self.closed = True

        def __getattr__(self, name):
            # Only methods the real desearch-py client has can be called.
            if name.startswith("_") or not callable(getattr(RealDesearch, name, None)):
                raise AttributeError(f"Desearch has no method {name!r}")

            async def method(*args, **kwargs):
                assert not args, f"{name} was called with positional arguments"
                rec.calls.append((name, kwargs))
                result = rec.responses[name]
                if isinstance(result, BaseException):
                    raise result
                return result

            return method

    monkeypatch.setattr(tools_module, "Desearch", FakeDesearch)
    return rec


@pytest.fixture
def spec(recorder):
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.poll_interval = 0
    return tool_spec


def _tools(tool_spec):
    return {tool.metadata.name: tool for tool in tool_spec.to_tool_list()}


def _plain(value):
    return tools_module._to_plain(value)


def _list_page(method):
    page = RESPONSES[method]
    return {"results": _plain(page.data), "next_cursor": page.metadata.next_cursor}


# (function, arguments, desearch-py method, its keyword arguments, expected output)
CALLS = [
    (
        "ai_search_tool",
        {"prompt": "What is Bittensor?", "tool": ["web", "twitter"]},
        "ai_search",
        {"prompt": "What is Bittensor?", "tools": ["web", "twitter"], "date_filter": None},
        RESPONSES["ai_search"],
    ),
    (
        "ai_web_links_search_tool",
        {"prompt": "Bittensor subnets", "count": 10},
        "ai_web_links_search",
        {"prompt": "Bittensor subnets", "tools": ["web"], "count": 10},
        _plain(RESPONSES["ai_web_links_search"]),
    ),
    (
        "ai_x_links_search_tool",
        {"prompt": "Bittensor news"},
        "ai_x_links_search",
        {"prompt": "Bittensor news", "count": None},
        _plain(RESPONSES["ai_x_links_search"]),
    ),
    (
        "search_api_tool",
        {
            "query": "Who won the 2024 Nobel Prize in Physics?",
            "mode": "fast",
            "count": 15,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "highlights": True,
            "page_text": False,
        },
        "search_api",
        {
            "query": "Who won the 2024 Nobel Prize in Physics?",
            "mode": "fast",
            "count": 15,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "highlights": True,
            "page_text": False,
        },
        _plain(RESPONSES["search_api"]),
    ),
    (
        "web_search_tool",
        {"query": "Bittensor", "start": 11},
        "web_search",
        {"query": "Bittensor", "start": 10},
        _plain(RESPONSES["web_search"])["data"],
    ),
    (
        "extract_tool",
        {"url": "https://example.com", "format": "html", "js": True, "wait": 500},
        "extract",
        {"url": "https://example.com", "format": "html", "js": True, "wait": 500},
        {"url": "https://example.com", "format": "html", "content": "Page text."},
    ),
    (
        "twitter_search_tool",
        {
            "query": "bittensor",
            "sort": "Latest",
            "count": 5,
            "user": "@opentensor",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "lang": "en",
            "verified": True,
            "blue_verified": False,
            "is_quote": False,
            "is_video": True,
            "is_image": False,
            "min_retweets": 1,
            "min_replies": 2,
            "min_likes": 3,
        },
        "x_search",
        {
            "query": "bittensor",
            "sort": "Latest",
            "count": 5,
            "user": "opentensor",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "lang": "en",
            "verified": True,
            "blue_verified": False,
            "is_quote": False,
            "is_video": True,
            "is_image": False,
            "min_retweets": 1,
            "min_replies": 2,
            "min_likes": 3,
        },
        [TWEET_DICT],
    ),
    (
        "x_post_by_id_tool",
        {"id": "1850000000000000000"},
        "x_post_by_id",
        {"id": "1850000000000000000"},
        TWEET_DICT,
    ),
    (
        "x_posts_by_urls_tool",
        {"urls": ["https://x.com/a/status/1", " https://x.com/b/status/2 "]},
        "x_posts_by_urls",
        {"urls": ["https://x.com/a/status/1", "https://x.com/b/status/2"]},
        [TWEET_DICT],
    ),
    (
        "x_posts_by_user_tool",
        {"user": "@elonmusk", "query": "AI", "count": 3},
        "x_posts_by_user",
        {"user": "elonmusk", "query": "AI", "count": 3},
        [TWEET_DICT],
    ),
    (
        "x_post_retweeters_tool",
        {"id": "1850000000000000000", "cursor": "rt-1"},
        "x_post_retweeters",
        {"id": "1850000000000000000", "cursor": "rt-1"},
        {"users": [{"id": "7", "username": "fan"}], "next_cursor": "rt-2"},
    ),
    (
        "x_user_posts_tool",
        {"username": "elonmusk", "cursor": "tl-1"},
        "x_user_posts",
        {"username": "elonmusk", "cursor": "tl-1"},
        _plain(RESPONSES["x_user_posts"]),
    ),
    (
        "x_user_replies_tool",
        {"user": "elonmusk", "count": 2, "query": "rocket"},
        "x_user_replies",
        {"user": "elonmusk", "count": 2, "query": "rocket"},
        [TWEET_DICT],
    ),
    (
        "x_post_replies_tool",
        {"post_id": "1850000000000000000", "count": 2},
        "x_post_replies",
        {"post_id": "1850000000000000000", "count": 2, "query": None},
        [TWEET_DICT],
    ),
    (
        "x_trends_tool",
        {"woeid": 23424977, "count": 30},
        "x_trends",
        {"woeid": 23424977, "count": 30},
        _plain(RESPONSES["x_trends"]),
    ),
    (
        "people_profiles_tool",
        {"platform": "linkedin", "has": ["email"], "min_score": 70, "cursor": "pp-2"},
        "list_people_profiles",
        {
            "platform": "linkedin",
            "sort": "score",
            "limit": 25,
            "cursor": "pp-2",
            "has": ["email"],
            "min_score": 70,
        },
        {"platform": "linkedin", "total": 0, "people": []},
    ),
]

# (function, action, arguments, desearch-py method, its keyword arguments)
SOCIAL_CALLS = [
    ("tiktok_tool", "search", {"query": "bittensor", "count": 2, "sort": "relevance"},
     "tiktok_search", {"query": "bittensor", "count": 2, "sort": "relevance"}),
    ("tiktok_tool", "post", {"id": "7620891413932166433"},
     "tiktok_post_by_id", {"id": "7620891413932166433"}),
    ("tiktok_tool", "profile", {"username": "@tiktok"},
     "tiktok_profile", {"username": "tiktok"}),
    ("tiktok_tool", "profile_posts", {"username": "tiktok", "count": 2},
     "tiktok_profile_posts", {"username": "tiktok", "count": 2}),
    ("tiktok_tool", "comments", {"id": "7620891413932166433", "count": 2},
     "tiktok_comments", {"video_id": "7620891413932166433", "count": 2}),
    ("tiktok_tool", "hashtag", {"tag": "#ai", "count": 2},
     "tiktok_hashtag", {"tag": "ai", "count": 2}),
    ("tiktok_tool", "trending", {"region": "GB", "count": 2},
     "tiktok_trending", {"region": "GB", "count": 2}),
    ("tiktok_tool", "trending", {},
     "tiktok_trending", {}),
    ("facebook_tool", "search", {"query": "running", "count": 2},
     "facebook_search", {"query": "running", "count": 2}),
    ("facebook_tool", "post", {"id": "1525856912919657"},
     "facebook_post_by_id", {"id": "1525856912919657"}),
    ("facebook_tool", "profile", {"profile_id": "facebook"},
     "facebook_profile", {"profile_id": "facebook"}),
    ("facebook_tool", "profile_posts", {"profile_id": "facebook", "count": 2},
     "facebook_profile_posts", {"profile_id": "facebook", "count": 2}),
    ("facebook_tool", "comments", {"id": "1525856912919657"},
     "facebook_comments", {"post_id": "1525856912919657"}),
    ("facebook_tool", "hashtag", {"tag": "ai", "count": 2},
     "facebook_hashtag", {"tag": "ai", "count": 2}),
    ("instagram_tool", "search", {"query": "instagram", "count": 2},
     "instagram_search", {"query": "instagram", "count": 2}),
    ("instagram_tool", "post", {"id": "https://www.instagram.com/p/DY79rTVS6OK/"},
     "instagram_media", {"id": "https://www.instagram.com/p/DY79rTVS6OK/"}),
    ("instagram_tool", "profile", {"username": "instagram"},
     "instagram_profile", {"username": "instagram"}),
    ("instagram_tool", "profile_posts", {"username": "@instagram", "count": 2},
     "instagram_profile_posts", {"username": "instagram", "count": 2}),
    ("instagram_tool", "comments", {"id": "DY79rTVS6OK", "count": 2},
     "instagram_comments", {"shortcode": "DY79rTVS6OK", "count": 2}),
    ("instagram_tool", "hashtag", {"tag": "ai"},
     "instagram_hashtag", {"tag": "ai"}),
]

PEOPLE_METHODS = {
    "x": "x_profile_search",
    "youtube": "youtube_creator_search",
    "linkedin": "linkedin_profile_search",
    "tiktok": "tiktok_creator_search",
}


@pytest.mark.parametrize(
    ("function", "arguments", "method", "sdk_kwargs", "expected"),
    CALLS,
    ids=[call[0] for call in CALLS],
)
def test_function_calls_the_right_sdk_method(
    spec, recorder, function, arguments, method, sdk_kwargs, expected
):
    output = _tools(spec)[function].call(**arguments)

    assert recorder.calls == [(method, sdk_kwargs)]
    assert output.raw_output == expected
    # A fresh client per call, with the spec's key, closed afterwards.
    (client,) = recorder.clients[1:]
    assert client.api_key == "test-key"
    assert client.closed


@pytest.mark.parametrize(
    ("function", "action", "arguments", "method", "sdk_kwargs"),
    SOCIAL_CALLS,
    ids=[f"{call[0]}-{call[1]}" for call in SOCIAL_CALLS],
)
def test_social_action_calls_the_right_sdk_method(
    spec, recorder, function, action, arguments, method, sdk_kwargs
):
    output = _tools(spec)[function].call(action=action, **arguments)

    ((called, kwargs),) = recorder.calls
    assert called == method
    if _SOCIAL_ACTIONS[function[: -len("_tool")]][action].returns_list:
        assert kwargs == {**sdk_kwargs, "include_metadata": True}
        assert output.raw_output == _list_page(method)
    else:
        assert kwargs == sdk_kwargs
        assert output.raw_output == _plain(RESPONSES[method])


def test_social_tools_cover_every_endpoint():
    covered = {(function, action) for function, action, *_ in SOCIAL_CALLS}
    expected = {
        (f"{platform}_tool", action)
        for platform, actions in _SOCIAL_ACTIONS.items()
        for action in actions
    }
    assert covered == expected
    assert len(expected) == 19
    for actions in _SOCIAL_ACTIONS.values():
        for action in actions.values():
            assert callable(getattr(RealDesearch, action.method, None)), action.method


@pytest.mark.parametrize(
    ("function", "action", "arguments", "method"),
    [
        ("tiktok_tool", "search", {"query": "ai"}, "tiktok_search"),
        ("tiktok_tool", "profile_posts", {"username": "tiktok"}, "tiktok_profile_posts"),
        ("tiktok_tool", "comments", {"id": "1"}, "tiktok_comments"),
        ("tiktok_tool", "hashtag", {"tag": "ai"}, "tiktok_hashtag"),
        ("facebook_tool", "search", {"query": "ai"}, "facebook_search"),
        ("facebook_tool", "profile_posts", {"profile_id": "facebook"}, "facebook_profile_posts"),
        ("facebook_tool", "comments", {"id": "1"}, "facebook_comments"),
        ("facebook_tool", "hashtag", {"tag": "ai"}, "facebook_hashtag"),
        ("instagram_tool", "profile_posts", {"username": "instagram"}, "instagram_profile_posts"),
        ("instagram_tool", "comments", {"id": "DY79rTVS6OK"}, "instagram_comments"),
        ("instagram_tool", "hashtag", {"tag": "ai"}, "instagram_hashtag"),
    ],
)
def test_cursor_is_passed_through_unchanged(
    spec, recorder, function, action, arguments, method
):
    cursor = "eyJwIjoyfQ==/+ opaque"
    recorder.responses[method] = _page([], cursor="next-" + cursor)

    result = getattr(spec, function)(action=action, cursor=cursor, **arguments)

    ((called, kwargs),) = recorder.calls
    assert called == method
    assert kwargs["cursor"] == cursor
    assert result == {"results": [], "next_cursor": "next-" + cursor}


def test_last_page_has_a_null_next_cursor(spec, recorder):
    recorder.responses["tiktok_hashtag"] = _page([TikTokPost(id="v1")])

    result = spec.tiktok_tool(action="hashtag", tag="ai", cursor="p3")

    assert [post["id"] for post in result["results"]] == ["v1"]
    assert result["next_cursor"] is None


def test_follow_a_cursor_page_by_page(spec, recorder):
    recorder.responses["instagram_profile_posts"] = _page([InstagramMedia(id="m1")], "c2")
    first = spec.instagram_tool(action="profile_posts", username="instagram", count=1)
    recorder.responses["instagram_profile_posts"] = _page([InstagramMedia(id="m2")])
    second = spec.instagram_tool(
        action="profile_posts", username="instagram", count=1, cursor=first["next_cursor"]
    )

    assert [post["id"] for post in first["results"]] == ["m1"]
    assert first["next_cursor"] == "c2"
    assert [post["id"] for post in second["results"]] == ["m2"]
    assert second["next_cursor"] is None
    assert [kwargs.get("cursor") for _, kwargs in recorder.calls] == [None, "c2"]


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        ("tiktok_tool", {"action": "likes", "id": "1"}, "Unknown action 'likes'. Use one of: search, post"),
        ("tiktok_tool", {"action": "search"}, "query (for action search) is required."),
        ("tiktok_tool", {"action": "post", "id": "  "}, "id (for action post) is required."),
        ("tiktok_tool", {"action": "profile", "username": "tiktok", "count": 5},
         "count does not apply to action 'profile'. It takes: username."),
        ("tiktok_tool", {"action": "trending", "cursor": "c2"},
         "cursor does not apply to action 'trending'. It takes: region, count."),
        ("tiktok_tool", {"action": "hashtag", "tag": "ai", "query": "x", "sort": "y"},
         "query, sort do not apply to action 'hashtag'."),
        ("tiktok_tool", {"action": "trending", "region": "USA"}, "region must be a two-letter"),
        ("tiktok_tool", {"action": "search", "query": "ai", "count": 0}, "count must be a whole number from 1 to 100."),
        ("tiktok_tool", {"action": "search", "query": "ai", "count": 101}, "count must be a whole number from 1 to 100."),
        ("tiktok_tool", {"action": "search", "query": "ai", "cursor": ""}, "cursor must be the next_cursor"),
        ("facebook_tool", {"action": "profile", "id": "facebook"},
         "profile_id (for action profile) is required."),
        ("facebook_tool", {"action": "trending"}, "Unknown action 'trending'"),
        ("instagram_tool", {"action": "search", "query": "ai", "cursor": "c2"},
         "cursor does not apply to action 'search'. It takes: query, count."),
        ("instagram_tool", {"action": "comments"}, "id (for action comments) is required."),
    ],
)
def test_social_arguments_are_checked_before_any_call(
    spec, recorder, function, arguments, message
):
    result = getattr(spec, function)(**arguments)

    assert isinstance(result, str)
    assert message in result
    assert recorder.calls == []


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        ("ai_search_tool", {"prompt": "x", "tool": []}, "tool must include at least one source."),
        ("ai_web_links_search_tool", {"prompt": " "}, "prompt is required."),
        ("ai_web_links_search_tool", {"prompt": "x", "count": 5}, "count must be a whole number from 10 to 200."),
        ("ai_x_links_search_tool", {"prompt": "x", "count": 201}, "count must be a whole number from 10 to 200."),
        ("search_api_tool", {"query": ""}, "query must be 1-2000 characters."),
        ("search_api_tool", {"query": "x" * 2001}, "query must be 1-2000 characters."),
        ("search_api_tool", {"query": "x", "count": 12}, "count must be 10, 15 or 20."),
        ("search_api_tool", {"query": "x", "mode": "slow"}, "mode must be fast or balanced."),
        ("search_api_tool", {"query": "x", "start_date": "01/02/2026"}, "start_date must be a date in YYYY-MM-DD format."),
        ("extract_tool", {"url": ""}, "url is required."),
        ("extract_tool", {"url": "https://example.com", "format": "pdf"}, "format must be text or html."),
        ("extract_tool", {"url": "https://example.com", "wait": 1000}, "wait only applies when js is true."),
        ("extract_tool", {"url": "https://example.com", "js": True, "wait": 30001}, "wait must be a whole number from 0 to 30000."),
        ("twitter_search_tool", {"query": "x", "count": 0}, "count must be a whole number from 1 to 100."),
        ("twitter_search_tool", {"query": "x", "end_date": "yesterday"}, "end_date must be a date in YYYY-MM-DD format."),
        ("x_post_by_id_tool", {"id": ""}, "id is required."),
        ("x_posts_by_urls_tool", {"urls": []}, "urls must be a list of one or more post URLs."),
        ("x_posts_by_user_tool", {"user": "elonmusk", "count": 500}, "count must be a whole number from 1 to 100."),
        ("x_post_retweeters_tool", {"id": None}, "id is required."),
        ("x_user_posts_tool", {"username": ""}, "username is required."),
        ("x_user_replies_tool", {"user": ""}, "user is required."),
        ("x_post_replies_tool", {"post_id": "1", "count": 0}, "count must be a whole number from 1 to 100."),
        ("x_trends_tool", {"count": 10}, "count must be a whole number from 30 to 100."),
        ("x_trends_tool", {"woeid": "worldwide"}, "woeid must be a whole number"),
        ("people_search_tool", {"query": "Indie hackers", "platform": "x", "count": 1}, "count must be between 5 and 200"),
        ("people_profiles_tool", {"platform": "x", "sort": "random"}, "sort must be one of: score, recent, audience."),
    ],
)
def test_arguments_are_checked_before_any_call(spec, recorder, function, arguments, message):
    result = getattr(spec, function)(**arguments)

    assert result == message or message in result
    assert recorder.calls == []


@pytest.mark.parametrize(("platform", "method"), sorted(PEOPLE_METHODS.items()))
def test_people_search_starts_the_platform_search_then_reads_results(
    spec, recorder, platform, method
):
    filters = {"x": {"verified_only": True}, "youtube": {"min_subscribers": 1000}}.get(
        platform, {"min_followers": 1000}
    )

    result = spec.people_search_tool(
        query="Founders who post about AI search",
        platform=platform,
        count=5,
        min_score=60,
        exclude=["someone"],
        filters=filters,
        idempotency_key="run-1",
        timeout=0,
    )

    assert recorder.calls[0] == (
        method,
        {
            "query": "Founders who post about AI search",
            "count": 5,
            "min_score": 60,
            "exclude": ["someone"],
            "filters": filters,
            "idempotency_key": "run-1",
        },
    )
    assert recorder.calls[1:] == [
        ("get_people_results", {"task_id": TASK.task_id, "cursor": None, "limit": 200})
    ]
    assert result["task_id"] == TASK.task_id
    assert result["people"] == []
    # One client for the whole search: start, poll and read share it.
    assert len(recorder.clients) == 2


def test_people_search_with_task_id_only_reads(spec, recorder):
    result = spec.people_search_tool(task_id=TASK.task_id, timeout=0)

    assert recorder.calls == [
        ("get_people_task", {"task_id": TASK.task_id}),
        ("get_people_results", {"task_id": TASK.task_id, "cursor": None, "limit": 200}),
    ]
    assert result["status"] == "completed"
    assert result["done"] is True


def test_people_search_polls_until_the_task_ends(spec, recorder):
    running = TASK.model_copy(update={"status": "running"})
    statuses = iter([running, running, TASK])
    recorder.responses["x_profile_search"] = running

    class Polls(dict):
        def __getitem__(self, name):
            if name == "get_people_task":
                return next(statuses)
            return super().__getitem__(name)

    recorder.responses = Polls(recorder.responses)

    result = spec.people_search_tool(query="Indie hackers", platform="x", timeout=30)

    assert recorder.methods() == [
        "x_profile_search",
        "get_people_task",
        "get_people_task",
        "get_people_task",
        "get_people_results",
    ]
    assert result["status"] == "completed"


def _valid_call(function):
    """One valid set of arguments for each spec function."""
    for name, arguments, *_ in CALLS:
        if name == function:
            return [arguments]
    if function == "people_search_tool":
        return [{"query": "Founders who post about AI", "platform": "x", "timeout": 0}]
    return [
        {"action": action, **arguments}
        for name, action, arguments, *_ in SOCIAL_CALLS
        if name == function
    ]


def test_no_function_calls_the_removed_sdk_methods(spec, recorder):
    """Regression: 0.0.2 called client.search, basic_twitter_search and basic_web_search,
    which no published desearch-py has, so every tool failed."""
    called = set()
    for function in DesearchToolSpec.spec_functions:
        calls = _valid_call(function)
        assert calls, f"no valid call for {function}"
        for arguments in calls:
            recorder.calls.clear()
            output = _tools(spec)[function].call(**arguments)
            assert not isinstance(output.raw_output, str), (function, output.raw_output)
            assert recorder.calls, function
            called.update(recorder.methods())

    assert not called & LEGACY_METHODS
    assert all(callable(getattr(RealDesearch, name, None)) for name in called)
    for legacy in LEGACY_METHODS:
        assert not hasattr(RealDesearch, legacy)


def test_source_never_names_the_removed_sdk_methods():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not attributes & LEGACY_METHODS, path.name
        # _call_desearch takes the method name as a string.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_call_desearch"
                and isinstance(node.args[1], ast.Constant)
            ):
                # The social tools pass _SOCIAL_ACTIONS methods, checked elsewhere.
                method = node.args[1]
                assert method.value not in LEGACY_METHODS
                assert callable(getattr(RealDesearch, method.value, None)), method.value


def test_every_spec_function_is_a_short_described_tool():
    tools = _tools(DesearchToolSpec(api_key="test-key"))

    assert list(tools) == DesearchToolSpec.spec_functions
    assert len(tools) == 20
    for name, tool in tools.items():
        # OpenAI rejects tool descriptions over 1024 characters.
        assert len(tool.metadata.description) <= 1024, name
        tool.metadata.to_openai_tool()
        assert tool.metadata.description.split("\n", 1)[1].strip(), f"{name} has no docstring"

    tiktok = tools["tiktok_tool"].metadata.get_parameters_dict()
    assert tiktok["properties"]["action"]["enum"] == list(_SOCIAL_ACTIONS["tiktok"])
    assert tiktok["required"] == ["action"]
    for name in ("tiktok_tool", "facebook_tool", "instagram_tool"):
        description = tools[name].metadata.description
        assert "next_cursor" in description
        for action in _SOCIAL_ACTIONS[name[: -len("_tool")]]:
            assert f"\n        {action} (" in description, (name, action)


def test_api_errors_come_back_as_text_without_the_key(spec, recorder):
    recorder.responses["tiktok_profile"] = DesearchAPIError(
        status=422, message="Unprocessable Entity", body='{"detail": "Bad cursor test-key"}'
    )

    result = spec.tiktok_tool(action="profile", username="tiktok")

    assert result == "HTTP 422: Bad cursor [REDACTED]"
    assert recorder.clients[-1].closed


def test_api_error_without_json_body_uses_the_text(spec, recorder):
    recorder.responses["extract"] = DesearchAPIError(
        status=502, message="Bad Gateway", body="upstream failed"
    )

    assert spec.extract_tool(url="https://example.com") == "HTTP 502: upstream failed"


def test_social_tool_works_from_an_async_agent(spec, recorder):
    output = asyncio.run(
        _tools(spec)["facebook_tool"].acall(action="hashtag", tag="ai", count=1)
    )

    assert output.raw_output == _list_page("facebook_hashtag")


def test_social_tool_works_inside_a_running_event_loop(spec, recorder):
    async def call_from_a_coroutine():
        return spec.instagram_tool(action="profile", username="instagram")

    assert asyncio.run(call_from_a_coroutine()) == _plain(RESPONSES["instagram_profile"])
