"""Company Search tool spec functions, tested against a local HTTP server (never the live API)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from llama_index_desearch.companies import compact_company, company_next_step
from llama_index_desearch.tools import DesearchToolSpec

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "0c9f3a52-7d1e-4f0a-9a3b-2b8e6f1d4c77"
TASK_PATH = f"/desearch/companies/tasks/{TASK_ID}"


def _task(status, result_count=0, **extra):
    return {
        "task_id": TASK_ID,
        "search_type": "company_search",
        "platform": "company",
        "status": status,
        "stage": None if status == "completed" else "searching",
        "query": "Series A fintech startups in Berlin",
        "count": 25,
        "result_count": result_count,
        "progress": {"candidates_found": 40},
        "errors": [],
        "completion_reason": "count_reached" if status == "completed" else None,
        "coverage_limited": False,
        "warnings": [],
        "metadata": {},
        "reserved_cost_usd": 2.25,
        "final_cost_usd": 0.18 if status == "completed" else None,
        "created_at": "2026-10-28T10:00:00Z",
        **extra,
    }


ACME = {
    "id": "company:acme.com",
    "platform": "company",
    "handle": "acme.com",
    "domain": "acme.com",
    "url": "https://acme.com/",
    "name": "Acme",
    "description": "Payment infrastructure for marketplaces.",
    "logo_url": "https://acme.com/logo.png",
    "location": {"country": "DE", "city": "Berlin", "address": None},
    "industry": "Fintech",
    "emails": [
        {
            "value": "hello@acme.com",
            "type": "general",
            "source": "website",
            "source_url": "https://acme.com/impressum",
            "person": None,
            "verification": {"status": "valid", "checks": {"mx": "ok"}},
        },
        {"value": "", "type": "other"},
    ],
    "phones": [
        {
            "value": "+493012345678",
            "type": "main",
            "source": "website",
            "source_url": "https://acme.com/impressum",
            "raw": "+49 30 1234 5678",
        }
    ],
    "socials": [
        {
            "type": "linkedin",
            "url": "https://www.linkedin.com/company/acme",
            "handle": "acme",
            "source_url": "https://acme.com/",
        },
        {"type": "x", "url": ""},
    ],
    "pages": [{"type": "home", "url": "https://acme.com/"}],
    "people": [
        {
            "name": "Anna Schmidt",
            "title": "Co-founder & CEO",
            "roles": ["ceo", "founder"],
            "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
            "x_url": None,
            "github_url": None,
            "emails": [
                {
                    "value": "anna.schmidt@acme.com",
                    "type": "personal",
                    "source": "inferred",
                    "person": "Anna Schmidt",
                    "confidence": "high",
                    "pattern": "first.last",
                    "verification": {"status": "valid"},
                }
            ],
            "phones": [],
            "evidence": {"quote": "Anna Schmidt, CEO", "source_url": "https://acme.com/impressum"},
            "verified": True,
        },
        {"title": "No name"},
    ],
    "primary_email": {
        "value": "anna.schmidt@acme.com",
        "type": "personal",
        "source": "inferred",
        "confidence": "high",
        "verification": {"status": "valid"},
    },
    "primary_contact": {
        "name": "Anna Schmidt",
        "title": "Co-founder & CEO",
        "email": "anna.schmidt@acme.com",
        "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
        "reason": "CEO, email inferred from the company's pattern and verified",
    },
    "contact_coverage": {"email": True, "phone": True, "linkedin": True},
    "enrichment": {
        "status": "complete",
        "stages": {"profile": "done", "contacts": "done", "people": "done"},
        "errors": [],
    },
    "score": {
        "value": 86,
        "summary": "Berlin payments infrastructure startup.",
        "reasons": ["Builds payment APIs", "Based in Berlin", "Raised a Series A", "Hiring"],
        "criteria_checks": [
            {"criterion": "Based in Berlin", "verdict": "met"},
            {"criterion": "Raised a Series A", "verdict": "unknown"},
            {"criterion": "Exclude consultancies", "verdict": "not_met", "is_disqualifier": True},
        ],
        "fully_verified": False,
    },
}

BETA = {
    "domain": "beta.io",
    "name": "Beta",
    "score": None,
    "enrichment": {"status": "partial", "stages": {"profile": "done", "people": "pending"}},
}


def _results(companies, has_more=False, cursor="c1", status="completed"):
    return {
        "task_id": TASK_ID,
        "status": status,
        "results": companies,
        "next_cursor": cursor,
        "has_more": has_more,
    }


class _FakeCompanyAPI:
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
def company_api():
    api = _FakeCompanyAPI()
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
def spec(company_api):
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.base_url = company_api.base_url
    tool_spec.poll_interval = 0.01
    return tool_spec


def _tools(tool_spec):
    return {tool.metadata.name: tool for tool in tool_spec.to_tool_list()}


def test_company_functions_are_listed_with_short_descriptions():
    tools = _tools(DesearchToolSpec(api_key="test-key"))
    for name in (
        "company_search_tool",
        "company_enrich_tool",
        "company_tasks_tool",
        "company_task_control_tool",
    ):
        # OpenAI rejects tool descriptions over 1024 characters.
        tools[name].metadata.to_openai_tool()

    for name in ("company_search_tool", "company_enrich_tool"):
        description = tools[name].metadata.description
        assert "BILLED" in description
        assert "task_id" in description
    assert "at least 2" in tools["company_search_tool"].metadata.description
    assert "at least 1" in tools["company_enrich_tool"].metadata.description
    assert "Free" in tools["company_tasks_tool"].metadata.description
    assert "cannot be undone" in tools["company_task_control_tool"].metadata.description

    params = tools["company_search_tool"].metadata.get_parameters_dict()["properties"]
    assert params["count"]["default"] == 25
    assert "leader_email" in params["require"]["description"]
    assert "managing_director" in params["roles"]["description"]
    control = tools["company_task_control_tool"].metadata.get_parameters_dict()
    assert control["properties"]["action"]["enum"] == ["pause", "resume", "cancel"]
    assert sorted(control["required"]) == ["action", "task_id"]


def test_company_search_does_not_require_unpublished_sdk():
    for name in ("pyproject.toml", "setup.py"):
        text = (ROOT / name).read_text()
        assert "desearch-py>=1.8" not in text
        assert "desearch-py>=1.0.0" in text


def test_company_search_tool_call_starts_waits_and_returns_compact_companies(
    spec, company_api
):
    company_api.add("POST", "/desearch/companies/search", 202, _task("queued", cost_usd=2.25))
    company_api.add("GET", TASK_PATH, 200, _task("running", 1, stage="enriching"))
    company_api.add("GET", TASK_PATH, 200, _task("completed", 2))
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME, BETA]))

    output = _tools(spec)["company_search_tool"].call(
        query="Series A fintech startups in Berlin",
        min_score=60,
        exclude=["stripe.com"],
        countries=["DE"],
        require=["email", "leader", "email"],
        roles=["ceo", "cto"],
        infer_emails=True,
        verify_emails=False,
        metadata={"campaign": "q4"},
        idempotency_key="run-1",
        timeout=30,
    )
    result = output.raw_output

    assert result["task_id"] == TASK_ID
    assert result["search_type"] == "company_search"
    assert result["status"] == "completed"
    assert result["done"] is True
    assert result["completion_reason"] == "count_reached"
    assert result["final_cost_usd"] == 0.18
    assert result["next_step"].startswith("Finished. These are all the companies")
    assert "1 company stayed partial" in result["next_step"]
    assert [company["domain"] for company in result["companies"]] == ["acme.com", "beta.io"]
    assert result["companies"][1]["score"] is None

    (create,) = company_api.calls("POST")
    assert create["path"] == "/desearch/companies/search"
    assert create["headers"]["Authorization"] == "test-key"
    assert create["headers"]["Idempotency-Key"] == "run-1"
    assert create["json"] == {
        "query": "Series A fintech startups in Berlin",
        "count": 25,
        "min_score": 60,
        "exclude": ["stripe.com"],
        "filters": {"countries": ["DE"], "require": ["email", "leader"]},
        "roles": ["ceo", "cto"],
        "infer_emails": True,
        "verify_emails": False,
        "metadata": {"campaign": "q4"},
    }
    assert len(company_api.calls("GET", TASK_PATH)) == 2
    (results_call,) = company_api.calls("GET", f"{TASK_PATH}/results")
    # No cursor: every company in its latest version.
    assert results_call["query"] == {"limit": "200"}
    assert all(r["headers"]["Authorization"] == "test-key" for r in company_api.requests)


def test_direct_python_call_uses_field_defaults(spec, company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("queued"))

    result = spec.company_search_tool(query="Dental clinics in Lisbon", timeout=0)

    assert result["status"] == "queued"
    assert result["companies"] == []
    (create,) = company_api.calls("POST")
    assert create["json"] == {"query": "Dental clinics in Lisbon", "count": 25}
    assert "Idempotency-Key" not in create["headers"]
    # Nothing delivered yet, so the results route is not read.
    assert company_api.calls("GET", f"{TASK_PATH}/results") == []


def test_compact_company_keeps_contacts_leaders_and_sources():
    company = compact_company(ACME)

    assert company["domain"] == "acme.com"
    assert company["location"] == {"country": "DE", "city": "Berlin"}
    assert company["score"] == 86
    assert company["why"] == "Berlin payments infrastructure startup."
    assert company["reasons"] == ["Builds payment APIs", "Based in Berlin", "Raised a Series A"]
    assert company["unverified_criteria"] == ["Raised a Series A (unknown)"]
    assert company["primary_email"] == {
        "value": "anna.schmidt@acme.com",
        "type": "personal",
        "source": "inferred",
        "confidence": "high",
        "verification": "valid",
    }
    assert company["emails"] == [
        {
            "value": "hello@acme.com",
            "type": "general",
            "source": "website",
            "source_url": "https://acme.com/impressum",
            "verification": "valid",
        }
    ]
    assert company["phones"] == [
        {"value": "+493012345678", "type": "main", "source_url": "https://acme.com/impressum"}
    ]
    assert company["socials"] == [
        {
            "type": "linkedin",
            "url": "https://www.linkedin.com/company/acme",
            "source_url": "https://acme.com/",
        }
    ]
    (leader,) = company["leaders"]
    assert leader["name"] == "Anna Schmidt"
    assert leader["roles"] == ["ceo", "founder"]
    assert leader["emails"][0]["pattern"] == "first.last"
    assert "person" not in leader["emails"][0]
    assert leader["evidence_url"] == "https://acme.com/impressum"
    assert company["enrichment"]["status"] == "complete"


def test_company_search_returns_companies_so_far_when_it_times_out(spec, company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("running", 1))
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([BETA], status="running"))

    result = spec.company_search_tool(query="Dental clinics in Lisbon", timeout=0)

    assert result["done"] is False
    assert [company["domain"] for company in result["companies"]] == ["beta.io"]
    assert f"task_id='{TASK_ID}'" in result["next_step"]
    assert "free" in result["next_step"]
    assert company_api.calls("GET", TASK_PATH) == []


def test_company_search_with_task_id_does_not_start_a_task(spec, company_api):
    company_api.add("GET", TASK_PATH, 200, _task("running", 1))
    company_api.add(
        "GET",
        TASK_PATH,
        200,
        _task(
            "completed",
            1,
            completion_reason="cost_limit",
            coverage_limited=True,
            warnings=[
                {"code": "ai_scoring_unavailable", "message": "AI scoring was unavailable."},
                {
                    "code": "linkedin_verification_unavailable",
                    "message": "Leaders found through LinkedIn search could not be checked.",
                },
            ],
        ),
    )
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME]))

    result = spec.company_search_tool(task_id=TASK_ID, timeout=30)

    assert company_api.calls("POST") == []
    assert result["coverage_limited"] is True
    assert [warning["code"] for warning in result["warnings"]] == [
        "ai_scoring_unavailable",
        "linkedin_verification_unavailable",
    ]
    assert result["next_step"].startswith("Finished early: a budget stopped it (cost_limit)")


def test_company_search_reads_every_results_page(spec, company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("completed", 2))
    company_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([ACME], has_more=True, cursor="p2")
    )
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([BETA], cursor="p3"))

    result = spec.company_search_tool(query="Series A fintech startups")

    assert [company["domain"] for company in result["companies"]] == ["acme.com", "beta.io"]
    pages = company_api.calls("GET", f"{TASK_PATH}/results")
    assert [page["query"] for page in pages] == [
        {"limit": "200"},
        {"limit": "200", "cursor": "p2"},
    ]


def test_idempotent_replay_is_marked(spec, company_api):
    company_api.add("POST", "/desearch/companies/search", 200, _task("completed"))

    result = spec.company_search_tool(
        query="Series A fintech startups", idempotency_key="same-key"
    )

    assert result["existing_task"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Set query to start a Company Search"),
        ({"query": "ab"}, "query must be 3-1000 characters"),
        ({"query": "Fintech startups", "count": 500}, "count must be between 5 and 200"),
        ({"query": "Fintech startups", "min_score": 101}, "min_score must be between 0 and 100"),
        ({"query": "Fintech startups", "require": ["website"]}, "Unknown require: website"),
        ({"query": "Fintech startups", "roles": ["intern"]}, "Unknown roles: intern"),
        ({"query": "Fintech startups", "countries": ["DE"] * 21}, "countries can list at most 20"),
        ({"query": "Fintech startups", "exclude": ["a.com"] * 1001}, "exclude can list at most 1000"),
        ({"query": "Fintech startups", "metadata": {"Campaign": "q4"}}, "metadata key 'Campaign'"),
        ({"query": "Fintech startups", "timeout": 5000}, "timeout must be"),
    ],
)
def test_invalid_searches_are_refused_before_any_call(spec, company_api, kwargs, message):
    result = spec.company_search_tool(**kwargs)

    assert isinstance(result, str)
    assert message in result
    assert company_api.requests == []


def test_company_enrich_sends_domains_and_returns_companies(spec, company_api):
    enrich_task = _task(
        "completed",
        1,
        search_type="company_enrich",
        platform="company_enrich",
        completion_reason="all_domains_processed",
    )
    company_api.add("POST", "/desearch/companies/enrich", 202, {**enrich_task, "status": "queued"})
    company_api.add("GET", TASK_PATH, 200, enrich_task)
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([{**ACME, "score": None}]))

    output = _tools(spec)["company_enrich_tool"].call(
        domains=["acme.com", " https://www.beta.io/about ", ""],
        roles=["ceo"],
        infer_emails=False,
        idempotency_key="enrich-1",
    )
    result = output.raw_output

    (create,) = company_api.calls("POST")
    assert create["path"] == "/desearch/companies/enrich"
    assert create["headers"]["Idempotency-Key"] == "enrich-1"
    assert create["json"] == {
        "domains": ["acme.com", "https://www.beta.io/about"],
        "roles": ["ceo"],
        "infer_emails": False,
    }
    assert result["search_type"] == "company_enrich"
    assert result["next_step"].startswith("Finished: it read every domain.")
    (company,) = result["companies"]
    assert company["score"] is None
    assert "why" not in company


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Set domains to start a Company Enrich"),
        ({"domains": [" "]}, "domains must list 1-200"),
        ({"domains": ["acme.com"] * 201}, "domains must list 1-200"),
        ({"domains": ["acme.com"], "roles": ["intern"]}, "Unknown roles: intern"),
    ],
)
def test_invalid_enrichments_are_refused_before_any_call(spec, company_api, kwargs, message):
    result = spec.company_enrich_tool(**kwargs)

    assert isinstance(result, str)
    assert message in result
    assert company_api.requests == []


def test_company_enrich_with_task_id_points_at_the_control_tool_when_paused(spec, company_api):
    company_api.add("GET", TASK_PATH, 200, _task("paused", 0, search_type="company_enrich"))

    result = spec.company_enrich_tool(task_id=TASK_ID, timeout=0)

    assert company_api.calls("POST") == []
    assert result["status"] == "paused"
    assert "company_task_control_tool" in result["next_step"]
    assert "action='resume'" in result["next_step"]


def test_company_tasks_sends_filters_and_returns_compact_tasks(spec, company_api):
    company_api.add(
        "GET",
        "/desearch/companies/tasks",
        200,
        {"tasks": [_task("completed", 3, metadata={"campaign": "q4"})], "next_cursor": "n2"},
    )

    output = _tools(spec)["company_tasks_tool"].call(
        type="enrich",
        status=["running", "paused"],
        limit=5,
        cursor="n1",
        metadata={"campaign": "q4"},
    )
    result = output.raw_output

    (call,) = company_api.calls("GET", "/desearch/companies/tasks")
    assert call["headers"]["Authorization"] == "test-key"
    assert call["query"] == {
        "type": "enrich",
        "status": "running,paused",
        "limit": "5",
        "cursor": "n1",
        "metadata.campaign": "q4",
    }
    (task,) = result["tasks"]
    assert task["metadata"] == {"campaign": "q4"}
    assert result["next_cursor"] == "n2"
    assert "cursor='n2'" in result["next_step"]


def test_company_tasks_direct_call_uses_defaults(spec, company_api):
    company_api.add("GET", "/desearch/companies/tasks", 200, {"tasks": [], "next_cursor": None})

    result = spec.company_tasks_tool()

    assert result == {"tasks": []}
    (call,) = company_api.calls("GET", "/desearch/companies/tasks")
    assert call["query"] == {"limit": "20"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": ["done"]}, "Unknown status: done"),
        ({"type": "people"}, "type must be one of"),
        ({"limit": 0}, "limit must be between 1 and 100"),
        ({"metadata": {"a": "1", "b": "2", "c": "3", "d": "4"}}, "at most 3 metadata labels"),
    ],
)
def test_company_tasks_refuses_bad_filters(spec, company_api, kwargs, message):
    result = spec.company_tasks_tool(**kwargs)

    assert message in result
    assert company_api.requests == []


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
def test_company_task_control_posts_the_action(spec, company_api, action):
    status = {"pause": "pausing", "resume": "running", "cancel": "cancelled"}[action]
    company_api.add("POST", f"{TASK_PATH}/{action}", 200, _task(status, 2))

    result = spec.company_task_control_tool(task_id=TASK_ID, action=action)

    (call,) = company_api.calls("POST")
    assert call["path"] == f"{TASK_PATH}/{action}"
    assert call["headers"]["Authorization"] == "test-key"
    assert result["status"] == status
    assert result["next_step"]


def test_company_task_control_conflict_keeps_the_api_detail(spec, company_api):
    company_api.add(
        "POST",
        f"{TASK_PATH}/resume",
        409,
        {"detail": "Only a paused task can be resumed.", "task": _task("completed", 2)},
    )

    result = spec.company_task_control_tool(task_id=TASK_ID, action="resume")

    assert result == (
        "HTTP 409: Only a paused task can be resumed. Task status: completed."
    )


def test_company_task_control_refuses_unknown_actions(spec, company_api):
    result = spec.company_task_control_tool(task_id=TASK_ID, action="delete")

    assert "action must be one of" in result
    assert company_api.requests == []


def test_http_errors_are_returned_without_the_api_key(spec, company_api):
    company_api.add(
        "POST",
        "/desearch/companies/search",
        402,
        {"detail": "Insufficient balance for this task."},
    )

    result = spec.company_search_tool(query="Series A fintech startups")

    assert result == "HTTP 402: Insufficient balance for this task."


def test_unreachable_api_is_returned_as_an_error():
    tool_spec = DesearchToolSpec(api_key="test-key")
    tool_spec.base_url = "http://127.0.0.1:9"

    result = tool_spec.company_tasks_tool()

    assert result.startswith("Could not reach Desearch")
    assert "test-key" not in result


@pytest.mark.parametrize(
    "task, expected",
    [
        ({"status": "completed", "completion_reason": "no_candidates"}, "found no company"),
        ({"status": "completed", "completion_reason": "zero_qualified"}, "none fit"),
        ({"status": "partial"}, "Finished with partial results"),
        ({"status": "failed"}, "not charged"),
        ({"status": "cancelled"}, "rest of the reservation is returned"),
        ({"status": "expired"}, "30 days"),
        ({"status": "running", "stage": "enriching"}, "Every company is delivered"),
    ],
)
def test_company_next_step(task, expected):
    step = company_next_step(task, TASK_ID, "company_search_tool", "company_task_control_tool")
    assert expected in step
