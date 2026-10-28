# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the latest release on PyPI is 0.0.2. Versions 0.0.3 and 0.1.0 were
> never released; their changes are part of 0.2.0.

## [0.2.0] - unreleased

> **Do not release yet.** 0.2.0 needs `desearch-py>=1.8.0`, which is not on
> PyPI yet (the latest published desearch-py is 1.2.1). Publish desearch-py
> 1.8.0 first. Until then `pip install llama-index-desearch==0.2.0` cannot
> resolve its dependencies, and the test workflow cannot install the package
> from PyPI either. The publish workflow uploads on every push to `main`, so
> merge this to `main` only after desearch-py 1.8.0 is published.

This release supersedes PR #2 (`fix/current-desearch-py-methods`, 0.0.3) and
PR #1 (`feat/people-search`, 0.1.0); both are merged into it.

### Fixed

- `ai_search_tool`, `twitter_search_tool` and `web_search_tool` failed on every
  call in 0.0.2: they called `client.search`, `client.basic_twitter_search` and
  `client.basic_web_search`, which no published desearch-py has. They now call
  `ai_search`, `x_search` and `web_search` (from PR #2).
- `ai_search_tool` no longer offers `youtube`: the API rejects it (HTTP 422
  when it is the only source) and drops it otherwise.
- Errors come back as one readable string for every function:
  `HTTP <status>: <detail>` for API errors and `Could not reach Desearch (...)`
  for connection errors. The API key is never part of it.

### Added

- **People Search** (from PR #1): `people_search_tool` (billed, waits for the
  search) and `people_profiles_tool` (free).
- **AI links search**: `ai_web_links_search_tool` and `ai_x_links_search_tool`.
- **Search API**: `search_api_tool` (`POST /search`), ranked pages each with the
  passage that best matches the query.
- **Extract**: `extract_tool` (`GET /web/extract`), returns
  `{"url", "format", "content"}`.
- **X (Twitter)**: `x_post_by_id_tool`, `x_posts_by_urls_tool`,
  `x_posts_by_user_tool`, `x_post_retweeters_tool`, `x_user_posts_tool`,
  `x_user_replies_tool`, `x_post_replies_tool` and `x_trends_tool`.
  `twitter_search_tool` takes every X search filter: `user`, `start_date`,
  `end_date`, `lang`, `verified`, `blue_verified`, `is_quote`, `is_video`,
  `is_image`, `min_retweets`, `min_replies` and `min_likes`.
- **TikTok, Facebook and Instagram**: `tiktok_tool`, `facebook_tool` and
  `instagram_tool`, one function per platform with an `action` argument that
  covers all 19 endpoints. Each action checks its arguments before any request
  (missing, unused or out-of-range arguments are refused). List actions return
  `{"results": [...], "next_cursor": ...}`; pass `next_cursor` back as `cursor`
  for the next page.
- `scripts/e2e-full-coverage.py`, a live smoke test of every function (needs
  `DESEARCH_API_KEY`; not run by CI).

### Changed

- Requires `desearch-py>=1.8.0,<2.0.0` and Python 3.10 or later.
- People Search runs on desearch-py's People Search methods instead of its own
  `requests` client, so `requests` is no longer a dependency. The tool output no
  longer has `existing_task`: desearch-py does not report whether an
  `Idempotency-Key` replayed an earlier search.
- Every function runs the async desearch-py client the same way: synchronously,
  with a new client per call on its own event loop, or on a worker thread when
  called inside a running loop. `FunctionTool.acall` runs it on a thread.
