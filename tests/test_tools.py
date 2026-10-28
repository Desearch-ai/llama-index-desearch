"""Spec functions tested end to end against a local HTTP server (never the live API).

The real desearch-py client sends each request; only its base URL points at the test server.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

import pytest

from llama_index_desearch.tools import DesearchToolSpec

AI_RESULT = {
    "text": "Bittensor is a decentralized network for machine learning.",
    "search": [
        {
            "title": "Bittensor",
            "link": "https://bittensor.com",
            "snippet": "Incentivized machine intelligence.",
        }
    ],
    "completion": "Bittensor rewards models that are useful to the network.",
}

TWEET = {
    "id": "1850000000000000000",
    "text": "Subnet 22 is live.",
    "url": "https://x.com/desearch_ai/status/1850000000000000000",
    "created_at": "2026-10-01T12:00:00Z",
    "reply_count": 3,
    "retweet_count": 5,
    "like_count": 40,
    "quote_count": 1,
    "bookmark_count": 2,
    "user": {"id": "42", "username": "desearch_ai", "name": "Desearch"},
}

WEB_PAGE = {
    "data": [
        {"title": "One", "link": "https://one.example", "snippet": "first"},
        {"title": "Two", "link": "https://two.example", "snippet": "second"},
        {"title": "Three", "link": "https://three.example", "snippet": "third"},
    ]
}


class _FakeDesearchAPI:
    """Records each request and answers with the payload added for its route."""

    def __init__(self):
        self.requests = []
        self.routes = {}
        self.lock = threading.Lock()
        self.base_url = ""

    def add(self, method, path, status, payload, headers=None):
        self.routes[(method, path)] = (status, payload, headers or {})

    def respond(self, method, path, query, headers, body):
        with self.lock:
            self.requests.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "headers": headers,
                    "json": body,
                }
            )
            return self.routes.get((method, path), (404, {"detail": "Not Found"}, {}))


def _handler(api):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            url = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            status, payload, headers = api.respond(
                self.command,
                url.path,
                dict(parse_qsl(url.query)),
                dict(self.headers),
                json.loads(raw) if raw else None,
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def desearch_api():
    api = _FakeDesearchAPI()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(api))
    api.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def spec(desearch_api):
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.client.base_url = desearch_api.base_url
    return tool_spec


def _tools(tool_spec):
    return {tool.metadata.name: tool for tool in tool_spec.to_tool_list()}


def test_search_functions_are_listed_with_short_descriptions():
    tools = _tools(DesearchToolSpec(api_key="test-key"))
    assert list(tools)[:3] == ["ai_search_tool", "twitter_search_tool", "web_search_tool"]
    for tool in tools.values():
        assert len(tool.metadata.description) <= 1024
        # OpenAI rejects tool descriptions over 1024 characters.
        tool.metadata.to_openai_tool()
        assert "Exa" not in tool.metadata.description

    params = tools["ai_search_tool"].metadata.get_parameters_dict()["properties"]
    assert "PAST_WEEK" in params["date_filter"]["description"]
    assert params["model"]["default"] == "NOVA"


def test_ai_search_tool_call_posts_prompt_sources_and_date_filter(spec, desearch_api):
    desearch_api.add("POST", "/desearch/ai/search", 200, AI_RESULT)

    output = _tools(spec)["ai_search_tool"].call(
        prompt="What is Bittensor?",
        tool=["web", "reddit"],
        model="ORBIT",
        date_filter="PAST_WEEK",
    )

    assert output.raw_output == AI_RESULT
    (request,) = desearch_api.requests
    assert request["method"] == "POST"
    assert request["path"] == "/desearch/ai/search"
    assert request["headers"]["Authorization"] == "test-key"
    # The current API has no model field.
    assert request["json"] == {
        "prompt": "What is Bittensor?",
        "tools": ["web", "reddit"],
        "date_filter": "PAST_WEEK",
        "streaming": False,
        "result_type": "LINKS_WITH_FINAL_SUMMARY",
    }


def test_ai_search_direct_call_sends_no_date_filter_by_default(spec, desearch_api):
    desearch_api.add("POST", "/desearch/ai/search", 200, AI_RESULT)

    result = spec.ai_search_tool(prompt="What is Bittensor?", tool=["web"])

    assert result == AI_RESULT
    (request,) = desearch_api.requests
    # desearch-py before 1.3 would send PAST_24_HOURS if date_filter were left out.
    assert "date_filter" not in request["json"]
    assert "model" not in request["json"]
    assert request["json"]["tools"] == ["web"]


def test_ai_search_refuses_an_empty_source_list(spec, desearch_api):
    result = spec.ai_search_tool(prompt="What is Bittensor?", tool=[])

    assert result == "tool must include at least one source."
    assert desearch_api.requests == []


def test_twitter_search_tool_call_uses_x_search_with_defaults(spec, desearch_api):
    desearch_api.add("GET", "/twitter", 200, [TWEET])

    output = _tools(spec)["twitter_search_tool"].call(query="Bittensor")

    # Plain dicts, without the fields the API left empty.
    assert output.raw_output == [TWEET]
    (request,) = desearch_api.requests
    assert request["path"] == "/twitter"
    assert request["headers"]["Authorization"] == "test-key"
    assert request["query"] == {"query": "Bittensor", "sort": "Top", "count": "10"}


def test_twitter_search_direct_call_forwards_sort_and_count(spec, desearch_api):
    desearch_api.add("GET", "/twitter", 200, [TWEET])

    result = spec.twitter_search_tool(query="from:desearch_ai", sort="Latest", count=5)

    assert result == [TWEET]
    assert desearch_api.requests[0]["query"] == {
        "query": "from:desearch_ai",
        "sort": "Latest",
        "count": "5",
    }


def test_web_search_maps_start_to_an_offset_and_trims_to_num(spec, desearch_api):
    desearch_api.add("GET", "/web", 200, WEB_PAGE)

    first_page = spec.web_search_tool(query="Bittensor subnet 22", num=2)
    next_page = spec.web_search_tool(query="Bittensor subnet 22", start=11)

    assert first_page == WEB_PAGE["data"][:2]
    assert next_page == WEB_PAGE["data"]
    assert [r["path"] for r in desearch_api.requests] == ["/web", "/web"]
    assert [r["query"] for r in desearch_api.requests] == [
        {"query": "Bittensor subnet 22", "start": "0"},
        {"query": "Bittensor subnet 22", "start": "10"},
    ]


def test_async_agent_call_runs_the_sync_function(spec, desearch_api):
    desearch_api.add("GET", "/web", 200, WEB_PAGE)

    output = asyncio.run(_tools(spec)["web_search_tool"].acall(query="Bittensor", num=1))

    assert output.raw_output == WEB_PAGE["data"][:1]


def test_direct_call_works_inside_a_running_event_loop(spec, desearch_api):
    desearch_api.add("GET", "/twitter", 200, [TWEET])

    async def call_from_a_coroutine():
        return spec.twitter_search_tool(query="Bittensor")

    assert asyncio.run(call_from_a_coroutine()) == [TWEET]


@pytest.mark.parametrize(
    ("function", "kwargs", "method", "path"),
    [
        ("ai_search_tool", {"prompt": "Bittensor", "tool": ["web"]}, "POST", "/desearch/ai/search"),
        ("twitter_search_tool", {"query": "Bittensor"}, "GET", "/twitter"),
        ("web_search_tool", {"query": "Bittensor"}, "GET", "/web"),
    ],
)
def test_http_errors_are_returned_as_text_without_the_api_key(
    spec, desearch_api, function, kwargs, method, path
):
    desearch_api.add(method, path, 401, {"detail": "Invalid API key"})

    result = getattr(spec, function)(**kwargs)

    assert result == "HTTP 401: Invalid API key"


def test_unreachable_api_is_returned_as_text():
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.client.base_url = "http://127.0.0.1:9"

    result = tool_spec.web_search_tool(query="Bittensor")

    assert result.startswith("Could not reach Desearch")
    assert "test-key" not in result


def test_social_list_action_returns_the_next_cursor_header(spec, desearch_api):
    desearch_api.add(
        "GET",
        "/desearch/tiktok/hashtag/ai",
        200,
        [{"id": "7620891413932166433", "text": "AI"}],
        headers={"x-desearch-next-cursor": "eyJwIjoyfQ=="},
    )

    first = spec.tiktok_tool(action="hashtag", tag="#ai", count=1)
    spec.tiktok_tool(action="hashtag", tag="ai", count=1, cursor=first["next_cursor"])

    assert first["next_cursor"] == "eyJwIjoyfQ=="
    assert [post["id"] for post in first["results"]] == ["7620891413932166433"]
    assert [r["query"] for r in desearch_api.requests] == [
        {"count": "1"},
        {"count": "1", "cursor": "eyJwIjoyfQ=="},
    ]
    assert desearch_api.requests[0]["headers"]["Authorization"] == "test-key"


def test_social_search_sends_the_cursor_in_the_body(spec, desearch_api):
    desearch_api.add("POST", "/desearch/facebook/search", 200, [])

    result = spec.facebook_tool(action="search", query="running", count=2, cursor="c2")

    # No x-desearch-next-cursor header: this was the last page.
    assert result == {"results": [], "next_cursor": None}
    assert desearch_api.requests[0]["json"] == {
        "query": "running",
        "count": 2,
        "cursor": "c2",
    }


def test_social_object_action_encodes_a_url_id_as_one_path_segment(spec, desearch_api):
    # The slashes stay encoded (%2F), so the URL is one path segment; yarl leaves ":" as is.
    path = "/desearch/instagram/media/https:%2F%2Fwww.instagram.com%2Fp%2FDY79rTVS6OK%2F"
    desearch_api.add("GET", path, 200, {"id": "1", "shortcode": "DY79rTVS6OK"})

    result = spec.instagram_tool(
        action="post", id="https://www.instagram.com/p/DY79rTVS6OK/"
    )

    assert result["shortcode"] == "DY79rTVS6OK"
    assert desearch_api.requests[0]["path"] == path


def test_extract_returns_the_page_text(spec, desearch_api):
    desearch_api.add("GET", "/web/extract", 200, "Example Domain")

    result = spec.extract_tool(url="https://example.com")

    assert result == {
        "url": "https://example.com",
        "format": "text",
        "content": '"Example Domain"',
    }
    assert desearch_api.requests[0]["query"] == {
        "url": "https://example.com",
        "format": "text",
        "js": "false",
    }
