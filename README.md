# LlamaIndex Tools Integration: Desearch

This tool connects to Desearch to enable your agent to perform searches across various platforms like web, Twitter, and more.

To begin, you need to obtain an API key from the Desearch developer dashboard.

## Usage

Here's an example usage of the `DesearchToolSpec`.

```python
# %pip install llama-index llama-index-core desearch-py

from llama_index_desearch.tools import DesearchToolSpec
from llama_index.agent.openai import OpenAIAgent

desearch_tool = DesearchToolSpec(
    api_key=os.environ["DESEARCH_API_KEY"],
)
agent = OpenAIAgent.from_tools(desearch_tool.to_tool_list())

agent.chat(
    "Can you find the latest news on quantum computing?"
)
```

### Available Functions

- **ai_search_tool**: Perform a search using Desearch across multiple platforms like web, Twitter, Reddit, etc.
- **twitter_search_tool**: Perform a basic Twitter search using the Desearch API.
- **web_search_tool**: Perform a basic web search using the Desearch API.

- **people_search_tool**: Find people on X, LinkedIn, YouTube or TikTok who match a plain-language description (Desearch People Search). Billed, and takes 1-5 minutes.
- **people_profiles_tool**: List the people your earlier People Searches found on one platform, with filters. Free.
- **company_search_tool**: Find companies that match a plain-language description, with public emails, phones, social profiles and leaders (Desearch Company Search). Billed, and runs for minutes.
- **company_enrich_tool**: Collect the same contact data for domains you already have (Desearch Company Enrich). Billed, and runs for minutes.
- **company_tasks_tool**: List your Company Search and Company Enrich tasks. Free.
- **company_task_control_tool**: Pause, resume or cancel a Company Search or Company Enrich task. Free.

This loader is designed to be used as a way to load data as a Tool in an Agent.

## People Search

[People Search](https://www.desearch.ai/docs/guide/apis/people-search) finds people on `x`, `linkedin`, `youtube` or `tiktok` from a plain-language description and returns them as compact profiles: name, handle, url, platform, audience, location, fit score, why they matched, contacts and links.

> **Billing and timing**: `people_search_tool` is billed. Starting a search reserves `count` profiles (5-200) at the route price; you pay only for the profiles it delivers, and a search that finds no one is free. A search runs in the background for 1-5 minutes. The tool waits up to `timeout` seconds (default 300), then returns the people found so far with the `task_id`. Call it again with only `task_id` to keep waiting: that is free and does not start a new search. `people_profiles_tool` is free and never starts a search.

```python
import os

from llama_index_desearch.tools import DesearchToolSpec

desearch_tool = DesearchToolSpec(api_key=os.environ["DESEARCH_API_KEY"])

# Start a billed search and wait up to 5 minutes for it.
people = desearch_tool.people_search_tool(
    query="Indie hackers building in public who share their MRR",
    platform="x",
    count=25,
    min_score=60,
    filters={"min_followers": 1000},
    timeout=300,
)
for person in people["people"]:
    print(person["handle"], person.get("audience"), person.get("why"))

# Still running? Keep waiting on the same search (free, nothing new is billed).
if not people["done"]:
    people = desearch_tool.people_search_tool(task_id=people["task_id"], timeout=300)

# Everyone your X searches found who has a public email (free).
library = desearch_tool.people_profiles_tool(
    platform="x", has=["email"], min_audience=1000, sort="score", limit=25
)
print(library["total"], library.get("next_cursor"))
```

Both functions return a `dict`, or an error string (for example `HTTP 402: ...`) like the other functions. Agents get them from `to_tool_list()` with the rest of the spec.

Platform filters go in `filters`:

| Platform | Filters |
| --- | --- |
| `x` | `min_followers`, `max_followers`, `verified_only`, `countries`, `keywords`, `exclude_keywords` |
| `linkedin` | `min_followers`, `max_followers`, `countries`, `keywords`, `exclude_keywords` |
| `youtube` | `min_subscribers`, `max_subscribers`, `countries`, `require_contact`, `keywords`, `exclude_keywords` |
| `tiktok` | `min_followers`, `max_followers`, `require_contact`, `keywords`, `exclude_keywords` |

A filter that does not apply to the platform is refused before any request is sent, so nothing is billed. Every search also takes `count`, `min_score`, `exclude` (handles to leave out) and `idempotency_key` (sending the same key again returns the first search instead of billing a new one).

`people_profiles_tool` filters: `q` (text in name, handle, bio or location), `has` (channels every person must have: `email`, `website`, `x`, `linkedin`, `instagram`, `youtube`, `tiktok`), `min_audience`, `max_audience`, `min_score`, `location`, `verified` (X and TikTok), plus `sort` (`score`, `recent`, `audience`), `limit` and `cursor`.

> People Search calls the `/desearch/people` routes over HTTP with `requests`, because the People Search methods in `desearch-py` arrive in 1.4.0, which is not on PyPI yet. The functions will switch to the SDK once it is published.

## Company Search

[Company Search](https://www.desearch.ai/docs/guide/apis/company-search) finds companies from a plain-language description, and Company Enrich reads the domains you already have. Both return compact companies: domain, name, url, location, fit score and why it matched (Company Search only; `score` is `None` for Company Enrich), primary email and contact, public emails, phones and social profiles with the page each came from, and leaders (CEO, founders, CTO and CFO by default) with their LinkedIn profiles and emails.

> **Billing and timing**: `company_search_tool` and `company_enrich_tool` are billed. Company Search reserves `count` companies (5-200) and Company Enrich one company per domain, at the route's per-company price ([prices](https://desearch.ai/pricing)); you pay only for the companies a task delivers (at least 2 for Company Search and 1 for Company Enrich when it delivers any), and a task that delivers none is free. A task runs in the background for minutes. The functions wait up to `timeout` seconds (default 300), then return the companies delivered so far with the `task_id`. Call either one again with only `task_id` to keep waiting: that is free and does not start a new task. `company_tasks_tool` and `company_task_control_tool` are free.

```python
import os

from llama_index_desearch.tools import DesearchToolSpec

desearch_tool = DesearchToolSpec(api_key=os.environ["DESEARCH_API_KEY"])

# Start a billed Company Search and wait up to 5 minutes for it.
companies = desearch_tool.company_search_tool(
    query="Series A fintech startups in Berlin building payment infrastructure",
    count=25,
    countries=["DE"],
    require=["email"],
    metadata={"campaign": "q4-outbound"},
    idempotency_key="q4-berlin-fintech",
    timeout=300,
)
for company in companies["companies"]:
    print(company["domain"], company.get("score"), company.get("primary_email"))

# Still running? Keep waiting on the same task (free, nothing new is billed).
if not companies["done"]:
    companies = desearch_tool.company_search_tool(task_id=companies["task_id"], timeout=300)

# Contacts and leaders for domains you already have (billed, one company per domain).
enriched = desearch_tool.company_enrich_tool(domains=["acme.com", "https://www.beta.io/about"])

# Your company tasks for one campaign, and pausing one (both free).
tasks = desearch_tool.company_tasks_tool(metadata={"campaign": "q4-outbound"})
desearch_tool.company_task_control_tool(task_id=companies["task_id"], action="pause")
```

`company_search_tool` also takes `min_score`, `exclude` (domains or URLs to leave out, up to 1000), `countries` (ISO codes or names, up to 20) and `require` (`email`, `phone`, `linkedin`, `leader`, `leader_email`: a company without all of them is not delivered or billed). Both start functions take `roles` (`ceo`, `founder`, `cto`, `cfo`, `coo`, `cmo`, `president`, `owner`, `managing_director`; default `ceo`, `founder`, `cto`, `cfo`), `infer_emails` and `verify_emails` (both default true), `metadata` (your own labels) and `idempotency_key`. Values the API would reject are refused before any request is sent, so nothing is billed.

`company_tasks_tool` filters: `type` (`search` or `enrich`), `status` (a list), `metadata` (up to 3 labels, exact match), plus `limit` and `cursor`. `company_task_control_tool` takes `task_id` and `action` (`pause`, `resume` or `cancel`; cancel cannot be undone).

Each company has `enrichment` with a `status` (`partial` while any stage, such as leaders, inferred emails or email checks, is still `pending`, then `complete`) and each stage. A stage cut by the task's time limit is `skipped`, with an `enrichment.errors` entry and the task's `enrichment_time_limit` warning. The functions read every company again on each call, so they return the latest version. A finished task has `completion_reason`, `coverage_limited` (when a budget stopped it before `count`) and `warnings`, such as `planning_fallback`, `ai_scoring_unavailable`, `enrichment_time_limit` or `linkedin_verification_unavailable`. A People Search `task_id` does not work on the company functions, and a company `task_id` does not work on `people_search_tool`.

> Company Search calls the `/desearch/companies` routes over HTTP with `requests`, because the Company Search methods in `desearch-py` arrive in 1.8.0, which is not on PyPI yet. The functions will switch to the SDK once it is published.

## Running Tests

The tests use a local HTTP server and never call the Desearch API:

```bash
pip install -e . pytest
pytest tests
```
