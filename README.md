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

## Running Tests

The tests use a local HTTP server and never call the Desearch API:

```bash
pip install -e . pytest
pytest tests
```
