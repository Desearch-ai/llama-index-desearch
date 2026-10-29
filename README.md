# LlamaIndex Tools Integration: Desearch

This tool connects to Desearch so your agent can search the web, X (Twitter), TikTok, Facebook and Instagram, read web pages, and find people.

To begin, you need to obtain an API key from the Desearch developer dashboard.

> 0.2.0 needs `desearch-py>=1.8.0`, which is not on PyPI yet. Until it is, install desearch-py 1.8.0 from source before this package.

## Usage

Here's an example usage of the `DesearchToolSpec`.

```python
# %pip install llama-index llama-index-core llama-index-desearch

import os

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

`to_tool_list()` returns every function below as a tool. To give an agent only some of them, pass their names: `desearch_tool.to_tool_list(spec_functions=["search_api_tool", "extract_tool"])`.

### Available Functions

| Function | What it does | Desearch API |
| --- | --- | --- |
| `ai_search_tool` | AI search of the web and X, or Hacker News, Reddit, Wikipedia and arXiv, with a summary | `POST /desearch/ai/search` |
| `ai_web_links_search_tool` | AI search that returns web links only | `POST /desearch/ai/search/links/web` |
| `ai_x_links_search_tool` | AI search that returns X posts only | `POST /desearch/ai/search/links/twitter` |
| `search_api_tool` | Ranked pages, each with the passage that best matches the query | `POST /search` |
| `web_search_tool` | Web search results (title, link, snippet) | `GET /web` |
| `extract_tool` | The text or HTML of a web page | `GET /web/extract` |
| `twitter_search_tool` | X search with author, date, language, media and engagement filters | `GET /twitter` |
| `x_post_by_id_tool` | One X post by ID | `GET /twitter/post` |
| `x_posts_by_urls_tool` | X posts by URL | `GET /twitter/urls` |
| `x_posts_by_user_tool` | One user's posts, optionally matching a query | `GET /twitter/post/user` |
| `x_post_retweeters_tool` | Users who reposted a post (paginated) | `GET /twitter/post/retweeters` |
| `x_user_posts_tool` | A user's timeline (paginated) | `GET /twitter/user/posts` |
| `x_user_replies_tool` | A user's posts and replies | `GET /twitter/replies` |
| `x_post_replies_tool` | Replies to a post | `GET /twitter/replies/post` |
| `x_trends_tool` | Trending topics for a location (WOEID) | `GET /twitter/trends` |
| `tiktok_tool` | TikTok search, video, profile, profile videos, comments, hashtag, trending | `/desearch/tiktok/*` (7 routes) |
| `facebook_tool` | Facebook Ad Library search, post, page, page posts, comments, hashtag | `/desearch/facebook/*` (6 routes) |
| `instagram_tool` | Instagram profile search, post, profile, profile posts, comments, hashtag | `/desearch/instagram/*` (6 routes) |
| `people_search_tool` | Find people on X, LinkedIn, YouTube or TikTok from a description. Billed; takes 1-5 minutes | `POST /desearch/people/search/{platform}` |
| `people_profiles_tool` | List the people your earlier People Searches found. Free | `GET /desearch/people/profiles` |

This loader is designed to be used as a way to load data as a Tool in an Agent.

### Return values and errors

Every function returns plain `dict`s and `list`s (JSON-ready). It never raises: on a failure it returns a string such as `HTTP 422: <detail from the API>` or `Could not reach Desearch (...)`, which an agent can read and act on. The API key never appears in it. Arguments are checked before any request is sent, so a bad argument is refused without being billed.

### Sync and async

desearch-py is an async client. Every function here is synchronous: each call opens its own desearch-py client, runs the request on its own event loop, and closes it. Inside a running event loop (a notebook cell, an async web handler) the request runs on a worker thread instead, so you can call the functions from anywhere. Async agents that call `FunctionTool.acall` get the same function run on a thread.

## TikTok, Facebook and Instagram

`tiktok_tool`, `facebook_tool` and `instagram_tool` each take an `action` and only the arguments that action uses. An argument the action does not use, a missing required one, or an out-of-range `count` (1-100) is refused before any request.

| Tool | `action` | Arguments | Paginated |
| --- | --- | --- | --- |
| `tiktok_tool` | `search` | `query`, `count`, `cursor`, `sort` | yes |
| | `post` | `id` (video ID) | |
| | `profile` | `username` | |
| | `profile_posts` | `username`, `count`, `cursor` | yes |
| | `comments` | `id` (video ID), `count`, `cursor` | yes |
| | `hashtag` | `tag`, `count`, `cursor` | yes |
| | `trending` | `region` (two letters, default `US`), `count` | no |
| `facebook_tool` | `search` | `query`, `count`, `cursor` (Meta Ad Library ads) | yes |
| | `post` | `id` | |
| | `profile` | `profile_id` (page ID, username or vanity handle) | |
| | `profile_posts` | `profile_id`, `count`, `cursor` | yes |
| | `comments` | `id` (post ID), `count`, `cursor` | yes |
| | `hashtag` | `tag`, `count`, `cursor` (Meta Ad Library ads) | yes |
| `instagram_tool` | `search` | `query`, `count` (profiles) | no |
| | `post` | `id` (shortcode, media ID or post URL) | |
| | `profile` | `username` | |
| | `profile_posts` | `username`, `count`, `cursor` | yes |
| | `comments` | `id` (the shortcode), `count`, `cursor` | yes |
| | `hashtag` | `tag`, `count`, `cursor` | yes |

A leading `@` on a username and `#` on a tag are removed. `post` and `profile` return one object. List actions return one page:

```python
{"results": [...], "next_cursor": "..."}  # next_cursor is None on the last page
```

To read the next page, call the same action with the same arguments and pass `next_cursor` back unchanged as `cursor`. A cursor is opaque and is valid for 24 hours; do not build or edit one.

```python
import os

from llama_index_desearch.tools import DesearchToolSpec

desearch_tool = DesearchToolSpec(api_key=os.environ["DESEARCH_API_KEY"])

posts = []
page = desearch_tool.tiktok_tool(action="hashtag", tag="ai", count=20)
while isinstance(page, dict):
    posts.extend(page["results"])
    if not page["next_cursor"] or len(posts) >= 60:
        break
    page = desearch_tool.tiktok_tool(
        action="hashtag", tag="ai", count=20, cursor=page["next_cursor"]
    )
if isinstance(page, str):
    print("Error:", page)
print(len(posts), "posts")
```

`x_post_retweeters_tool` and `x_user_posts_tool` paginate the same way, with `next_cursor` in the returned `dict`.

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

## Running Tests

The tests never call the Desearch API: some send real requests to a local HTTP server, the rest replace the desearch-py client with a mock.

```bash
pip install -e . pytest
pytest tests
```

### Live smoke test

`scripts/e2e-full-coverage.py` calls every function once against the real API with the smallest counts and prints `PASS name` or `FAIL name: error` per call. It exits with status 1 if any call fails. The calls are billed at normal prices (a few cents in total). It never starts a People Search unless you pass `--with-people`, which starts one TikTok search (`count=5`, billed) and cancels it.

```bash
DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py
DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py --base-url https://api.desearch.ai --with-people
```
