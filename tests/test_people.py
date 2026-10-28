"""People Search tool spec functions, tested against a local HTTP server (never the live API)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from llama_index_desearch.people import compact_person
from llama_index_desearch.tools import DesearchToolSpec

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "5b33f3ee-168b-4780-9c8e-63a626781872"
TASK_PATH = f"/desearch/people/tasks/{TASK_ID}"


def _task(status, result_count=0, **extra):
    return {
        "task_id": TASK_ID,
        "search_type": "x_profile_search",
        "platform": "x",
        "status": status,
        "stage": None if status == "completed" else "searching",
        "query": "Indie hackers building in public",
        "count": 25,
        "result_count": result_count,
        "progress": {"candidates": 40},
        "errors": [],
        "reserved_cost_usd": 1.0,
        "final_cost_usd": 0.2 if status == "completed" else None,
        "created_at": "2026-10-19T10:00:00Z",
        **extra,
    }


PERSON_X = {
    "id": "x:1",
    "platform": "x",
    "handle": "levelsio",
    "url": "https://x.com/levelsio",
    "name": "Pieter Levels",
    "bio": "Building in public",
    "location": "Portugal",
    "metrics": {"followers": 600000},
    "attributes": {"verified": True},
    "links": [{"type": "website", "url": "https://levels.io"}],
    "contacts": [{"type": "email", "value": "pieter@example.com"}],
    "evidence": [{"type": "post", "url": "https://x.com/levelsio/status/1"}],
    "score": {"value": 92, "summary": "Shares MRR updates every month."},
}

PERSON_X_2 = {
    "id": "x:2",
    "platform": "x",
    "handle": "marc_louvion",
    "metrics": {"followers": 120000},
    "score": {"value": 81, "reasons": ["Shares revenue screenshots"]},
}


def _results(people, has_more=False, cursor="c1", status="completed"):
    return {
        "task_id": TASK_ID,
        "status": status,
        "results": people,
        "next_cursor": cursor,
        "has_more": has_more,
    }


class _FakePeopleAPI:
    """Records each request and answers from scripted responses.

    A route keeps returning its last response, so ``add`` a sequence for
    polling (running, running, completed) and the last one sticks.
    """

    def __init__(self):
        self.requests = []
        self.routes = {}
        self.lock = threading.Lock()
        self.base_url = ""

    def add(self, method, path, status, payload):
        self.routes.setdefault((method, path), []).append((status, payload))

    def calls(self, method=None, path=None):
        return [
            r
            for r in self.requests
            if (method is None or r["method"] == method)
            and (path is None or r["path"] == path)
        ]

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
            queue = self.routes.get((method, path))
            if not queue:
                return 404, {"detail": "Not Found"}
            return queue.pop(0) if len(queue) > 1 else queue[0]


def _handler(api):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            url = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            status, payload = api.respond(
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
            self.end_headers()
            self.wfile.write(data)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def people_api():
    api = _FakePeopleAPI()
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
def spec(people_api):
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.base_url = people_api.base_url
    tool_spec.poll_interval = 0.01
    return tool_spec


def _tools(tool_spec):
    return {tool.metadata.name: tool for tool in tool_spec.to_tool_list()}


def test_people_functions_are_listed_with_short_descriptions():
    tools = _tools(DesearchToolSpec(api_key="test-key"))
    assert list(tools) == [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
        "people_search_tool",
        "people_profiles_tool",
    ]
    for tool in tools.values():
        # OpenAI rejects tool descriptions over 1024 characters.
        tool.metadata.to_openai_tool()

    search = tools["people_search_tool"].metadata.description
    assert "BILLED" in search
    assert "1-5 minutes" in search
    assert "task_id" in search
    assert "Free" in tools["people_profiles_tool"].metadata.description

    params = tools["people_search_tool"].metadata.get_parameters_dict()["properties"]
    assert params["platform"]["anyOf"][0]["enum"] == ["x", "youtube", "linkedin", "tiktok"]
    assert params["count"]["default"] == 25
    assert "verified_only" in params["filters"]["description"]


def test_package_metadata():
    pyproject = (ROOT / "pyproject.toml").read_text()
    setup_py = (ROOT / "setup.py").read_text()
    assert 'version = "0.1.0"' in pyproject
    assert 'version="0.1.0"' in setup_py
    for text in (pyproject, setup_py):
        # People Search must not need the unpublished desearch-py 1.4.0.
        assert "desearch-py>=1.1.0,<2.0.0" in text
        assert "desearch-py>=1.4" not in text
        assert "requests>=" in text


def test_people_search_tool_call_starts_waits_and_returns_compact_people(
    spec, people_api
):
    people_api.add("POST", "/desearch/people/search/x", 202, _task("queued"))
    people_api.add("GET", TASK_PATH, 200, _task("running", 1))
    people_api.add("GET", TASK_PATH, 200, _task("completed", 2))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X, PERSON_X_2]))

    output = _tools(spec)["people_search_tool"].call(
        query="Indie hackers building in public",
        platform="x",
        min_score=60,
        exclude=["someone"],
        filters={"min_followers": 1000, "verified_only": True},
        idempotency_key="run-1",
        timeout=30,
    )
    result = output.raw_output

    assert result["task_id"] == TASK_ID
    assert result["status"] == "completed"
    assert result["done"] is True
    assert result["final_cost_usd"] == 0.2
    assert result["next_step"].startswith("Finished.")
    assert result["people"] == [
        {
            "name": "Pieter Levels",
            "handle": "levelsio",
            "platform": "x",
            "url": "https://x.com/levelsio",
            "audience": 600000,
            "location": "Portugal",
            "verified": True,
            "score": 92,
            "why": "Shares MRR updates every month.",
            "contacts": [{"type": "email", "value": "pieter@example.com"}],
            "links": [{"type": "website", "url": "https://levels.io"}],
            "evidence_url": "https://x.com/levelsio/status/1",
        },
        {
            "handle": "marc_louvion",
            "platform": "x",
            "audience": 120000,
            "score": 81,
            "why": "Shares revenue screenshots",
        },
    ]

    (create,) = people_api.calls("POST")
    assert create["path"] == "/desearch/people/search/x"
    assert create["headers"]["Authorization"] == "test-key"
    assert create["headers"]["Idempotency-Key"] == "run-1"
    assert create["json"] == {
        "query": "Indie hackers building in public",
        "count": 25,
        "min_score": 60,
        "exclude": ["someone"],
        "filters": {"min_followers": 1000, "verified_only": True},
    }
    assert len(people_api.calls("GET", TASK_PATH)) == 2
    assert all(r["headers"]["Authorization"] == "test-key" for r in people_api.requests)


def test_direct_python_call_uses_field_defaults(spec, people_api):
    people_api.add("POST", "/desearch/people/search/tiktok", 202, _task("queued"))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([], status="queued"))

    result = spec.people_search_tool(
        query="Cooking creators who share recipes", platform="tiktok", timeout=0
    )

    assert result["status"] == "queued"
    (create,) = people_api.calls("POST")
    assert create["json"] == {"query": "Cooking creators who share recipes", "count": 25}
    assert "Idempotency-Key" not in create["headers"]


def test_people_search_returns_people_so_far_when_it_times_out(spec, people_api):
    people_api.add("POST", "/desearch/people/search/linkedin", 202, _task("queued"))
    people_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([PERSON_X], status="running")
    )

    result = spec.people_search_tool(
        query="Heads of sales at fintech startups", platform="linkedin", timeout=0
    )

    assert result["done"] is False
    assert [person["handle"] for person in result["people"]] == ["levelsio"]
    assert f"task_id='{TASK_ID}'" in result["next_step"]
    assert "free" in result["next_step"]
    assert people_api.calls("GET", TASK_PATH) == []


def test_people_search_with_task_id_does_not_start_a_search(spec, people_api):
    people_api.add("GET", TASK_PATH, 200, _task("running", 1))
    people_api.add("GET", TASK_PATH, 200, _task("partial", 1, errors=["x timeout"]))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X_2]))

    result = spec.people_search_tool(task_id=TASK_ID, timeout=30)

    assert people_api.calls("POST") == []
    assert result["status"] == "partial"
    assert result["errors"] == ["x timeout"]
    assert [person["handle"] for person in result["people"]] == ["marc_louvion"]


def test_people_search_reads_every_results_page(spec, people_api):
    people_api.add("POST", "/desearch/people/search/x", 202, _task("completed", 2))
    people_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([PERSON_X], has_more=True, cursor="p2")
    )
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X_2], cursor="p3"))

    result = spec.people_search_tool(query="Indie hackers", platform="x")

    assert [person["handle"] for person in result["people"]] == [
        "levelsio",
        "marc_louvion",
    ]
    pages = people_api.calls("GET", f"{TASK_PATH}/results")
    assert [page["query"] for page in pages] == [
        {"limit": "200"},
        {"limit": "200", "cursor": "p2"},
    ]


def test_idempotent_replay_is_marked(spec, people_api):
    people_api.add("POST", "/desearch/people/search/x", 200, _task("completed"))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([]))

    result = spec.people_search_tool(
        query="Indie hackers", platform="x", idempotency_key="same-key"
    )

    assert result["existing_task"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"query": "Heads of sales", "platform": "linkedin", "filters": {"verified_only": True}},
            "verified_only does not apply to linkedin",
        ),
        (
            {"query": "Cooking creators", "platform": "youtube", "filters": {"min_followers": 5}},
            "min_followers does not apply to youtube",
        ),
        ({"query": "Indie hackers"}, "Set query and platform"),
        ({"platform": "x"}, "Set query and platform"),
        ({"query": "ab", "platform": "x"}, "query must be 3-1000 characters"),
        ({"query": "Indie hackers", "platform": "x", "count": 500}, "count must be between 5 and 200"),
        ({"query": "Indie hackers", "platform": "facebook"}, "Unknown platform"),
        ({"query": "Indie hackers", "platform": "x", "timeout": 5000}, "timeout must be"),
    ],
)
def test_invalid_searches_are_refused_before_any_call(spec, people_api, kwargs, message):
    result = spec.people_search_tool(**kwargs)

    assert isinstance(result, str)
    assert message in result
    assert people_api.requests == []


def test_http_errors_are_returned_without_the_api_key(spec, people_api):
    people_api.add(
        "POST",
        "/desearch/people/search/x",
        402,
        {"detail": "Insufficient balance for this search."},
    )

    result = spec.people_search_tool(query="Indie hackers", platform="x")

    assert result == "HTTP 402: Insufficient balance for this search."


def test_unreachable_api_is_returned_as_an_error(people_api):
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.base_url = "http://127.0.0.1:9"

    result = tool_spec.people_profiles_tool(platform="x")

    assert result.startswith("Could not reach Desearch")
    assert "test-key" not in result


def test_people_profiles_sends_filters_and_returns_compact_people(spec, people_api):
    profile = {
        **PERSON_X,
        "platform": "tiktok",
        "found_by": {
            "task_id": TASK_ID,
            "query": "Cooking creators",
            "found_at": "2026-10-19T10:05:00Z",
            "times_found": 2,
        },
    }
    people_api.add(
        "GET",
        "/desearch/people/profiles",
        200,
        {"profiles": [profile], "total": 31, "next_cursor": "next-page"},
    )

    output = _tools(spec)["people_profiles_tool"].call(
        platform="tiktok",
        q="recipes",
        sort="audience",
        limit=10,
        has=["email", "instagram", "email"],
        min_audience=1000,
        max_audience=500000,
        min_score=70,
        location="united states",
        verified=True,
    )
    result = output.raw_output

    (call,) = people_api.calls("GET", "/desearch/people/profiles")
    assert call["headers"]["Authorization"] == "test-key"
    assert call["query"] == {
        "platform": "tiktok",
        "q": "recipes",
        "sort": "audience",
        "limit": "10",
        "has": "email,instagram",
        "min_audience": "1000",
        "max_audience": "500000",
        "min_score": "70",
        "location": "united states",
        "verified": "true",
    }
    assert result["total"] == 31
    assert result["next_cursor"] == "next-page"
    assert "cursor='next-page'" in result["next_step"]
    (person,) = result["people"]
    assert person["handle"] == "levelsio"
    assert person["found_by"]["times_found"] == 2


def test_people_profiles_direct_call_uses_defaults(spec, people_api):
    people_api.add(
        "GET",
        "/desearch/people/profiles",
        200,
        {"profiles": [], "total": 0, "next_cursor": None},
    )

    result = spec.people_profiles_tool(platform="x")

    assert result == {"platform": "x", "total": 0, "people": []}
    (call,) = people_api.calls("GET", "/desearch/people/profiles")
    assert call["query"] == {"platform": "x", "sort": "score", "limit": "25"}


def test_people_profiles_rejects_unknown_channels(spec, people_api):
    result = spec.people_profiles_tool(platform="x", has=["phone"])

    assert "Unknown has channel(s): phone" in result
    assert people_api.requests == []


def test_compact_person_uses_subscribers_and_linkedin_fields():
    youtube = compact_person(
        {
            "platform": "youtube",
            "handle": "@chef",
            "metrics": {"subscribers": 54000},
            "score": {"value": 77.5, "summary": "Posts weekly recipes."},
        }
    )
    assert youtube["audience"] == 54000
    assert youtube["why"] == "Posts weekly recipes."

    linkedin = compact_person(
        {
            "platform": "linkedin",
            "handle": "jane-doe",
            "attributes": {"headline": "VP Sales", "company": "Acme"},
        }
    )
    assert linkedin["headline"] == "VP Sales"
    assert linkedin["company"] == "Acme"
