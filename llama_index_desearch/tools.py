import asyncio
import concurrent.futures
import json
import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import aiohttp
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from desearch_py import Desearch, DesearchAPIError

from llama_index_desearch.people import (
    list_people_profiles,
    people_search_body,
    search_people,
)


class TwitterScraperMedia(BaseModel):
    media_url: str = ""
    type: str = ""


class TwitterScraperUser(BaseModel):
    # Available in both, scraped and api based tweets.
    id: Optional[str] = Field(example="123456789")
    url: Optional[str] = Field(example="https://x.com/example_user")
    name: Optional[str] = Field(example="John Doe")
    username: Optional[str] = Field(example="johndoe")
    created_at: Optional[str] = Field(example="2023-01-01T00:00:00Z")

    # Only available in scraped tweets
    description: Optional[str] = Field(example="This is an example user description.")
    favourites_count: Optional[int] = Field(example=100)
    followers_count: Optional[int] = Field(example=1500)
    listed_count: Optional[int] = Field(example=10)
    media_count: Optional[int] = Field(example=50)
    profile_image_url: Optional[str] = Field(example="https://example.com/profile.jpg")
    statuses_count: Optional[int] = Field(example=500)
    verified: Optional[bool] = Field(example=True)


class BasicTwitterSearchResponse(BaseModel):
    # Available in both, scraped and api based tweets.
    user: Optional[TwitterScraperUser]
    id: Optional[str] = Field(example="987654321")
    text: Optional[str] = Field(example="This is an example tweet.")
    reply_count: Optional[int] = Field(example=10)
    retweet_count: Optional[int] = Field(example=5)
    like_count: Optional[int] = Field(example=100)
    view_count: Optional[int] = Field(example=1000)
    quote_count: Optional[int] = Field(example=2)
    impression_count: Optional[int] = Field(example=1500)
    bookmark_count: Optional[int] = Field(example=3)
    url: Optional[str] = Field(example="https://x.com/example_tweet")
    created_at: Optional[str] = Field(example="2023-01-01T00:00:00Z")
    media: Optional[List[TwitterScraperMedia]] = Field(default_factory=list, example=[])

    # Only available in scraped tweets
    is_quote_tweet: Optional[bool] = Field(example=False)
    is_retweet: Optional[bool] = Field(example=False)


class WebSearchResult(BaseModel):
    title: str = Field(
        ..., description="EXCLUSIVE Major coffee buyers face losses as Colombia ..."
    )
    snippet: str = Field(
        ...,
        description="Coffee farmers in Colombia, the world's No. 2 arabica producer, have failed to deliver up to 1 million bags of beans this year or nearly 10% ...",
    )
    link: str = Field(
        ...,
        description="https://www.reuters.com/world/americas/exclusive-major-coffee-buyers-face-losses-colombia-farmers-fail-deliver-2021-10-11/",
    )
    date: Optional[str] = Field(
        None, description="21 hours ago"
    )  # Optional, as it might not always be present
    source: str = Field(..., description="Reuters")

    author: Optional[str] = Field(None, description="Reuters")

    image: Optional[str] = Field(
        None,
        description="https://static.reuters.com/resources/2021/10/11/Reuters/Reuters_20211011_0000_01.jpg?w=800&h=533&q=80&crop=1",
    )
    favicon: Optional[str] = Field(
        None,
        description="https://static.reuters.com/resources/2021/10/11/Reuters/Reuters_20211011_0000_01.jpg?w=800&h=533&q=80&crop=1",
    )
    highlights: Optional[List[str]] = Field(
        None, description="List of highlights as strings."
    )


PeoplePlatform = Literal["x", "youtube", "linkedin", "tiktok"]
PeopleChannel = Literal[
    "email", "website", "x", "linkedin", "instagram", "youtube", "tiktok"
]

TikTokAction = Literal[
    "search", "post", "profile", "profile_posts", "comments", "hashtag", "trending"
]
FacebookAction = Literal[
    "search", "post", "profile", "profile_posts", "comments", "hashtag"
]
InstagramAction = Literal[
    "search", "post", "profile", "profile_posts", "comments", "hashtag"
]


class _SocialAction(NamedTuple):
    """One action of a social tool: the desearch-py method it calls and its arguments."""

    method: str
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()
    # Tool argument name -> desearch-py argument name, where they differ.
    renames: Tuple[Tuple[str, str], ...] = ()
    # True when the route returns a list of items (one page).
    returns_list: bool = False


# Every TikTok, Facebook and Instagram route, by tool and action. Only actions
# with "cursor" in ``optional`` paginate.
_SOCIAL_ACTIONS: Dict[str, Dict[str, _SocialAction]] = {
    "tiktok": {
        "search": _SocialAction(
            "tiktok_search", ("query",), ("count", "cursor", "sort"), returns_list=True
        ),
        "post": _SocialAction("tiktok_post_by_id", ("id",)),
        "profile": _SocialAction("tiktok_profile", ("username",)),
        "profile_posts": _SocialAction(
            "tiktok_profile_posts", ("username",), ("count", "cursor"), returns_list=True
        ),
        "comments": _SocialAction(
            "tiktok_comments",
            ("id",),
            ("count", "cursor"),
            renames=(("id", "video_id"),),
            returns_list=True,
        ),
        "hashtag": _SocialAction(
            "tiktok_hashtag", ("tag",), ("count", "cursor"), returns_list=True
        ),
        "trending": _SocialAction(
            "tiktok_trending", (), ("region", "count"), returns_list=True
        ),
    },
    "facebook": {
        "search": _SocialAction(
            "facebook_search", ("query",), ("count", "cursor"), returns_list=True
        ),
        "post": _SocialAction("facebook_post_by_id", ("id",)),
        "profile": _SocialAction("facebook_profile", ("profile_id",)),
        "profile_posts": _SocialAction(
            "facebook_profile_posts",
            ("profile_id",),
            ("count", "cursor"),
            returns_list=True,
        ),
        "comments": _SocialAction(
            "facebook_comments",
            ("id",),
            ("count", "cursor"),
            renames=(("id", "post_id"),),
            returns_list=True,
        ),
        "hashtag": _SocialAction(
            "facebook_hashtag", ("tag",), ("count", "cursor"), returns_list=True
        ),
    },
    "instagram": {
        # Instagram search returns one page; the API rejects a cursor there.
        "search": _SocialAction(
            "instagram_search", ("query",), ("count",), returns_list=True
        ),
        "post": _SocialAction("instagram_media", ("id",)),
        "profile": _SocialAction("instagram_profile", ("username",)),
        "profile_posts": _SocialAction(
            "instagram_profile_posts",
            ("username",),
            ("count", "cursor"),
            returns_list=True,
        ),
        "comments": _SocialAction(
            "instagram_comments",
            ("id",),
            ("count", "cursor"),
            renames=(("id", "shortcode"),),
            returns_list=True,
        ),
        "hashtag": _SocialAction(
            "instagram_hashtag", ("tag",), ("count", "cursor"), returns_list=True
        ),
    },
}

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _arg(value: Any) -> Any:
    """Use a Field's default when a spec function is called directly in Python."""
    if isinstance(value, FieldInfo):
        default = value.get_default(call_default_factory=True)
        return None if default is PydanticUndefined else default
    return value


def _to_plain(value: Any) -> Any:
    """Turn desearch-py's pydantic models into plain dicts and lists, without empty fields."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _run_desearch(client: Desearch, work: Callable[[Desearch], Awaitable[Any]]) -> Any:
    """Run ``work`` (a coroutine function of a desearch-py client) from a sync spec function.

    Every spec function runs the async desearch-py client this way. Each call
    opens its own client with the key and base URL of ``client``, so its
    aiohttp session never outlives the event loop it ran on.
    """

    async def call() -> Any:
        request_client = Desearch(api_key=client.api_key, base_url=client.base_url)
        try:
            return await work(request_client)
        finally:
            await request_client.close()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call())
    # Called from inside a running event loop (for example a notebook cell):
    # run the request on a worker thread's own loop instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, call()).result()


def _call_desearch(client: Desearch, method: str, **kwargs: Any) -> Any:
    """Call one desearch-py method and return its result as plain dicts and lists."""
    return _to_plain(
        _run_desearch(client, lambda c: getattr(c, method)(**kwargs))
    )


def _api_error_detail(error: DesearchAPIError) -> str:
    body = error.body or ""
    try:
        data = json.loads(body) if body else None
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get("detail") is not None:
        detail = data["detail"]
        if not isinstance(detail, str):
            detail = json.dumps(detail)
        task = data.get("task")
        if isinstance(task, dict) and isinstance(task.get("status"), str):
            return f"{detail} Task status: {task['status']}."
        return detail
    if body:
        return body if len(body) <= 500 else f"{body[:500]}..."
    return error.message or "Request failed."


def _error_text(error: BaseException, api_key: str) -> str:
    """What a spec function returns instead of raising. Never holds the API key."""
    if isinstance(error, DesearchAPIError):
        text = f"HTTP {error.status}: {_api_error_detail(error)}"
    elif isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError)):
        reason = str(error)
        text = f"Could not reach Desearch ({type(error).__name__}" + (
            f": {reason})" if reason else ")"
        )
    else:
        text = str(error) or type(error).__name__
    return text.replace(api_key, "[REDACTED]") if api_key else text


def _check_range(name: str, value: Any, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}.")


def _required_text(name: str, value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required.")
    return value.strip()


def _check_date(name: str, value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not _DATE.match(value)):
        raise ValueError(f"{name} must be a date in YYYY-MM-DD format.")


class DesearchToolSpec(BaseToolSpec):
    """Desearch tool spec: AI search, web, X (Twitter), TikTok, Facebook, Instagram and People Search.

    Every function is synchronous and returns plain dicts and lists, or an
    error string (``HTTP <status>: <detail>``) instead of raising. It runs the
    async desearch-py client on its own event loop, or on a worker thread when
    it is called inside a running loop. ``FunctionTool.acall`` runs it on a
    thread, so async agents work too.
    """

    spec_functions = [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
        "people_search_tool",
        "people_profiles_tool",
        "ai_web_links_search_tool",
        "ai_x_links_search_tool",
        "search_api_tool",
        "extract_tool",
        "x_post_by_id_tool",
        "x_posts_by_urls_tool",
        "x_posts_by_user_tool",
        "x_post_retweeters_tool",
        "x_user_posts_tool",
        "x_user_replies_tool",
        "x_post_replies_tool",
        "x_trends_tool",
        "tiktok_tool",
        "facebook_tool",
        "instagram_tool",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""

        self.client = Desearch(api_key=api_key)
        # Seconds between People Search status checks.
        self.poll_interval = 5.0

    def _error(self, error: BaseException) -> str:
        return _error_text(error, self.client.api_key)

    # ─── AI search ───────────────────────────────────────────────────────────

    def ai_search_tool(
        self,
        prompt: str = Field(description="The search prompt or query."),
        tool: List[
            Literal[
                "web",
                "twitter",
                "hackernews",
                "reddit",
                "wikipedia",
                "arxiv",
            ]
        ] = Field(
            description=(
                "Sources to search; at least one. web and twitter (X) are searched "
                "directly; hackernews, reddit, wikipedia and arxiv limit the web "
                "search to that site."
            )
        ),
        model: str = Field(
            default="NOVA",
            description=(
                "Kept for compatibility: 'NOVA', 'ORBIT' or 'HORIZON'. The current "
                "Desearch API picks the model itself and ignores this value."
            ),
        ),
        date_filter: Optional[str] = Field(
            default=None,
            description=(
                "Only include results from this window: PAST_24_HOURS, PAST_2_DAYS, "
                "PAST_WEEK, PAST_2_WEEKS, PAST_MONTH, PAST_2_MONTHS, PAST_YEAR or "
                "PAST_2_YEARS. Leave empty for no date filter."
            ),
        ),
    ) -> str | dict:
        """
        Search the web and X (Twitter), or Hacker News, Reddit, Wikipedia and arXiv, with Desearch AI search.

        Returns:
            str | dict: The results per source and a summary, or an error string.
        """
        try:
            prompt, tool, date_filter = map(_arg, (prompt, tool, date_filter))
            if not tool:
                raise ValueError("tool must include at least one source.")
            # desearch-py has no model argument.
            return _call_desearch(
                self.client,
                "ai_search",
                prompt=prompt,
                tools=tool,
                date_filter=date_filter,
            )
        except Exception as e:
            return self._error(e)

    def ai_web_links_search_tool(
        self,
        prompt: str = Field(description="What to search the web for."),
        count: Optional[int] = Field(
            default=None, description="Number of links to return (10-200; default 10)."
        ),
    ) -> Union[dict, str]:
        """
        Find web page links for a prompt with Desearch AI search, without a written summary.

        Returns:
            dict | str: Links with title and snippet, or an error string.
        """
        try:
            prompt, count = map(_arg, (prompt, count))
            prompt = _required_text("prompt", prompt)
            _check_range("count", count, 10, 200)
            # Web Search is the only source this route supports.
            return _call_desearch(
                self.client,
                "ai_web_links_search",
                prompt=prompt,
                tools=["web"],
                count=count,
            )
        except Exception as e:
            return self._error(e)

    def ai_x_links_search_tool(
        self,
        prompt: str = Field(description="What to search X (Twitter) for."),
        count: Optional[int] = Field(
            default=None, description="Number of posts to return (10-200; default 10)."
        ),
    ) -> Union[dict, str]:
        """
        Find X (Twitter) posts relevant to a prompt with Desearch AI search, without a written summary.

        Returns:
            dict | str: Matching posts with their links, or an error string.
        """
        try:
            prompt, count = map(_arg, (prompt, count))
            prompt = _required_text("prompt", prompt)
            _check_range("count", count, 10, 200)
            return _call_desearch(
                self.client, "ai_x_links_search", prompt=prompt, count=count
            )
        except Exception as e:
            return self._error(e)

    def search_api_tool(
        self,
        query: str = Field(description="Search query (1-2000 characters)."),
        mode: Optional[Literal["fast", "balanced"]] = Field(
            default=None,
            description="fast returns sooner; balanced (the default) favors relevance.",
        ),
        count: Optional[int] = Field(
            default=None, description="Number of results: 10 (default), 15 or 20."
        ),
        start_date: Optional[str] = Field(
            default=None,
            description="Only pages published on or after this date (YYYY-MM-DD).",
        ),
        end_date: Optional[str] = Field(
            default=None,
            description="Only pages published on or before this date (YYYY-MM-DD).",
        ),
        highlights: Optional[bool] = Field(
            default=None,
            description="Include the passage that best matches the query (default true).",
        ),
        page_text: Optional[bool] = Field(
            default=None, description="Include the full page text (default false)."
        ),
    ) -> Union[dict, str]:
        """
        Search the web and get ranked pages, each with the passage that best answers the query (Desearch Search API).

        Use it to ground an answer in sources. Set page_text=true only when you need whole pages.

        Returns:
            dict | str: results (url, title, published_date, highlights, page_text) and cost, or an error string.
        """
        try:
            query, mode, count, start_date = map(_arg, (query, mode, count, start_date))
            end_date, highlights, page_text = map(_arg, (end_date, highlights, page_text))
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
                raise ValueError("query must be 1-2000 characters.")
            if mode is not None and mode not in ("fast", "balanced"):
                raise ValueError("mode must be fast or balanced.")
            if count is not None and count not in (10, 15, 20):
                raise ValueError("count must be 10, 15 or 20.")
            _check_date("start_date", start_date)
            _check_date("end_date", end_date)
            return _call_desearch(
                self.client,
                "search_api",
                query=query,
                mode=mode,
                count=count,
                start_date=start_date,
                end_date=end_date,
                highlights=highlights,
                page_text=page_text,
            )
        except Exception as e:
            return self._error(e)

    # ─── Web ─────────────────────────────────────────────────────────────────

    def web_search_tool(
        self,
        query: str = Field(description="The search query."),
        num: int = Field(
            default=10,
            description="Maximum number of results to return (one page has up to 10).",
        ),
        start: int = Field(
            default=1, description="1-based position of the first result, for paging."
        ),
    ) -> str | List[dict]:
        """
        Search the web with Desearch.

        Args:
            query (str): The search query.
            num (int, optional): Maximum number of results to return. Defaults to 10.
            start (int, optional): 1-based position of the first result. Defaults to 1.

        Returns:
            str | List[dict]: Results with title, link and snippet, or an error string.
        """
        try:
            query, num, start = map(_arg, (query, num, start))
            # web_search takes a 0-based skip offset and returns one page of
            # results; it has no result-count argument.
            page = _call_desearch(
                self.client, "web_search", query=query, start=max(start - 1, 0)
            )
            return page["data"][:num]
        except Exception as e:
            return self._error(e)

    def extract_tool(
        self,
        url: str = Field(description="Public URL of the page to read."),
        format: Literal["text", "html"] = Field(
            default="text", description="text (default) or html."
        ),
        js: bool = Field(
            default=False,
            description=(
                "Render the page in a browser first, for pages built by JavaScript. "
                "Billed at a higher rate."
            ),
        ),
        wait: Optional[int] = Field(
            default=None,
            description="Extra milliseconds to wait after the page loads (0-30000). Needs js=true.",
        ),
    ) -> Union[dict, str]:
        """
        Read the content of a web page by URL (Desearch Extract).

        Returns:
            dict | str: {"url", "format", "content"} with the page text or HTML, or an error string.
        """
        try:
            url, format, js, wait = map(_arg, (url, format, js, wait))
            url = _required_text("url", url)
            if format not in ("text", "html"):
                raise ValueError("format must be text or html.")
            _check_range("wait", wait, 0, 30000)
            if wait is not None and not js:
                raise ValueError("wait only applies when js is true.")
            content = _call_desearch(
                self.client, "extract", url=url, format=format, js=bool(js), wait=wait
            )
            return {"url": url, "format": format, "content": content}
        except Exception as e:
            return self._error(e)

    # ─── X (Twitter) ─────────────────────────────────────────────────────────

    def twitter_search_tool(
        self,
        query: str = Field(
            description="The X (Twitter) search query; X advanced search operators work."
        ),
        sort: str = Field(default="Top", description="Sort order: 'Top' or 'Latest'."),
        count: int = Field(
            default=10, description="Number of posts to return (1-100)."
        ),
        user: Optional[str] = Field(
            default=None, description="Only posts by this username (without @)."
        ),
        start_date: Optional[str] = Field(
            default=None, description="Only posts on or after this date (YYYY-MM-DD, UTC)."
        ),
        end_date: Optional[str] = Field(
            default=None, description="Only posts on or before this date (YYYY-MM-DD, UTC)."
        ),
        lang: Optional[str] = Field(
            default=None, description="Language code, e.g. en, es, fr."
        ),
        verified: Optional[bool] = Field(
            default=None, description="Only posts by verified users."
        ),
        blue_verified: Optional[bool] = Field(
            default=None, description="Only posts by blue-check (X Premium) users."
        ),
        is_quote: Optional[bool] = Field(default=None, description="Only quote posts."),
        is_video: Optional[bool] = Field(
            default=None, description="Only posts with a video."
        ),
        is_image: Optional[bool] = Field(
            default=None, description="Only posts with an image."
        ),
        min_retweets: Optional[int] = Field(
            default=None, description="Minimum number of reposts."
        ),
        min_replies: Optional[int] = Field(
            default=None, description="Minimum number of replies."
        ),
        min_likes: Optional[int] = Field(
            default=None, description="Minimum number of likes."
        ),
    ) -> str | List[dict]:
        """
        Search posts on X (Twitter) with Desearch, with optional author, date, language, media and engagement filters.

        Returns:
            str | List[dict]: Posts with their author and engagement counts, or an error string.
        """
        try:
            query, sort, count, user = map(_arg, (query, sort, count, user))
            start_date, end_date, lang = map(_arg, (start_date, end_date, lang))
            verified, blue_verified = map(_arg, (verified, blue_verified))
            is_quote, is_video, is_image = map(_arg, (is_quote, is_video, is_image))
            min_retweets, min_replies, min_likes = map(
                _arg, (min_retweets, min_replies, min_likes)
            )
            _check_range("count", count, 1, 100)
            _check_date("start_date", start_date)
            _check_date("end_date", end_date)
            return _call_desearch(
                self.client,
                "x_search",
                query=query,
                sort=sort,
                count=count,
                user=user.lstrip("@") if isinstance(user, str) else user,
                start_date=start_date,
                end_date=end_date,
                lang=lang,
                verified=verified,
                blue_verified=blue_verified,
                is_quote=is_quote,
                is_video=is_video,
                is_image=is_image,
                min_retweets=min_retweets,
                min_replies=min_replies,
                min_likes=min_likes,
            )
        except Exception as e:
            return self._error(e)

    def x_post_by_id_tool(
        self,
        id: str = Field(description="The post ID, the number at the end of its URL."),
    ) -> Union[dict, str]:
        """
        Get one X (Twitter) post by its ID, with its author and engagement counts.

        Returns:
            dict | str: The post, or an error string.
        """
        try:
            id = _required_text("id", _arg(id))
            return _call_desearch(self.client, "x_post_by_id", id=id)
        except Exception as e:
            return self._error(e)

    def x_posts_by_urls_tool(
        self,
        urls: List[str] = Field(
            description="Post URLs, e.g. ['https://x.com/user/status/1892527552029499853']."
        ),
    ) -> Union[List[dict], str]:
        """
        Get full X (Twitter) posts for a list of post URLs.

        Returns:
            List[dict] | str: The posts, or an error string.
        """
        try:
            urls = _arg(urls)
            if isinstance(urls, str):
                urls = [urls]
            if not urls or not all(isinstance(u, str) and u.strip() for u in urls):
                raise ValueError("urls must be a list of one or more post URLs.")
            return _call_desearch(
                self.client, "x_posts_by_urls", urls=[u.strip() for u in urls]
            )
        except Exception as e:
            return self._error(e)

    def x_posts_by_user_tool(
        self,
        user: str = Field(description="Username, without @."),
        query: Optional[str] = Field(
            default=None, description="Only this user's posts that match this query."
        ),
        count: Optional[int] = Field(
            default=None, description="Number of posts to return (1-100; default 10)."
        ),
    ) -> Union[List[dict], str]:
        """
        Search one X (Twitter) user's posts, optionally only those matching a query.

        Returns:
            List[dict] | str: The posts, or an error string.
        """
        try:
            user, query, count = map(_arg, (user, query, count))
            user = _required_text("user", user).lstrip("@")
            _check_range("count", count, 1, 100)
            return _call_desearch(
                self.client, "x_posts_by_user", user=user, query=query, count=count
            )
        except Exception as e:
            return self._error(e)

    def x_post_retweeters_tool(
        self,
        id: str = Field(description="ID of the post."),
        cursor: Optional[str] = Field(
            default=None,
            description="next_cursor from the previous page; leave empty for the first page.",
        ),
    ) -> Union[dict, str]:
        """
        List the users who reposted (retweeted) an X post, one page at a time.

        Returns:
            dict | str: {"users": [...], "next_cursor"}; pass next_cursor back as cursor for the next page. Or an error string.
        """
        try:
            id, cursor = map(_arg, (id, cursor))
            id = _required_text("id", id)
            return _call_desearch(self.client, "x_post_retweeters", id=id, cursor=cursor)
        except Exception as e:
            return self._error(e)

    def x_user_posts_tool(
        self,
        username: str = Field(description="Username, without @."),
        cursor: Optional[str] = Field(
            default=None,
            description="next_cursor from the previous page; leave empty for the first page.",
        ),
    ) -> Union[dict, str]:
        """
        Read an X (Twitter) user's timeline, newest first, one page at a time.

        Returns:
            dict | str: {"user", "tweets": [...], "next_cursor"}; pass next_cursor back as cursor for the next page. Or an error string.
        """
        try:
            username, cursor = map(_arg, (username, cursor))
            username = _required_text("username", username).lstrip("@")
            return _call_desearch(
                self.client, "x_user_posts", username=username, cursor=cursor
            )
        except Exception as e:
            return self._error(e)

    def x_user_replies_tool(
        self,
        user: str = Field(description="Username, without @."),
        count: Optional[int] = Field(
            default=None, description="Number of posts to return (1-100; default 10)."
        ),
        query: Optional[str] = Field(
            default=None, description="Only posts and replies that match this query."
        ),
    ) -> Union[List[dict], str]:
        """
        Get the posts and replies an X (Twitter) user wrote.

        Returns:
            List[dict] | str: The posts and replies, or an error string.
        """
        try:
            user, count, query = map(_arg, (user, count, query))
            user = _required_text("user", user).lstrip("@")
            _check_range("count", count, 1, 100)
            return _call_desearch(
                self.client, "x_user_replies", user=user, count=count, query=query
            )
        except Exception as e:
            return self._error(e)

    def x_post_replies_tool(
        self,
        post_id: str = Field(description="ID of the post."),
        count: Optional[int] = Field(
            default=None, description="Number of replies to return (1-100; default 10)."
        ),
        query: Optional[str] = Field(
            default=None, description="Only replies that match this query."
        ),
    ) -> Union[List[dict], str]:
        """
        Get the replies to an X (Twitter) post.

        Returns:
            List[dict] | str: The replies, or an error string.
        """
        try:
            post_id, count, query = map(_arg, (post_id, count, query))
            post_id = _required_text("post_id", post_id)
            _check_range("count", count, 1, 100)
            return _call_desearch(
                self.client, "x_post_replies", post_id=post_id, count=count, query=query
            )
        except Exception as e:
            return self._error(e)

    def x_trends_tool(
        self,
        woeid: int = Field(
            default=1,
            description=(
                "WOEID of the location: 1 is worldwide (default), 23424977 the United "
                "States, 23424975 the United Kingdom."
            ),
        ),
        count: Optional[int] = Field(
            default=None, description="Number of trends to return (30-100; default 30)."
        ),
    ) -> Union[dict, str]:
        """
        Get the trending topics on X (Twitter) for a location.

        Returns:
            dict | str: {"trends": [{"name", "query", "rank"}], "woeid"}, or an error string.
        """
        try:
            woeid, count = map(_arg, (woeid, count))
            if isinstance(woeid, bool) or not isinstance(woeid, int):
                raise ValueError("woeid must be a whole number, e.g. 1 for worldwide.")
            _check_range("count", count, 30, 100)
            return _call_desearch(self.client, "x_trends", woeid=woeid, count=count)
        except Exception as e:
            return self._error(e)

    # ─── TikTok, Facebook and Instagram ──────────────────────────────────────

    def _social(self, platform: str, action: Any, **arguments: Any) -> Union[dict, str]:
        """Check the arguments of one social action, call it and shape the result."""
        try:
            action = _arg(action)
            values = {name: _arg(value) for name, value in arguments.items()}
            actions = _SOCIAL_ACTIONS[platform]
            spec = actions.get(action)
            if spec is None:
                raise ValueError(
                    f"Unknown action {action!r}. Use one of: {', '.join(actions)}."
                )
            accepted = spec.required + spec.optional
            for name in spec.required:
                values[name] = _required_text(f"{name} (for action {action})", values[name])
            unused = [n for n, v in values.items() if v is not None and n not in accepted]
            if unused:
                verb = "does" if len(unused) == 1 else "do"
                takes = ", ".join(accepted) or "no other arguments"
                raise ValueError(
                    f"{', '.join(unused)} {verb} not apply to action {action!r}. "
                    f"It takes: {takes}."
                )
            _check_range("count", values.get("count"), 1, 100)
            cursor = values.get("cursor")
            if cursor is not None and (not isinstance(cursor, str) or not cursor.strip()):
                raise ValueError(
                    "cursor must be the next_cursor from the previous page, unchanged."
                )
            region = values.get("region")
            if region is not None and (
                not isinstance(region, str) or not re.fullmatch(r"[A-Za-z]{2}", region)
            ):
                raise ValueError("region must be a two-letter country code, e.g. US.")
            if values.get("username"):
                values["username"] = values["username"].lstrip("@")
            if values.get("tag"):
                values["tag"] = values["tag"].lstrip("#")

            renames = dict(spec.renames)
            kwargs = {
                renames.get(name, name): values[name]
                for name in accepted
                if values.get(name) is not None
            }
            if not spec.returns_list:
                return _call_desearch(self.client, spec.method, **kwargs)
            # include_metadata=True returns the x-desearch-next-cursor header too.
            page = _run_desearch(
                self.client,
                lambda c: getattr(c, spec.method)(**kwargs, include_metadata=True),
            )
            return {
                "results": _to_plain(page.data),
                "next_cursor": page.metadata.next_cursor,
            }
        except Exception as e:
            return self._error(e)

    def tiktok_tool(
        self,
        action: TikTokAction = Field(description="What to read; see the tool description."),
        query: Optional[str] = Field(default=None, description="search: keywords."),
        id: Optional[str] = Field(
            default=None, description="post and comments: the video ID."
        ),
        username: Optional[str] = Field(
            default=None, description="profile and profile_posts: username, without @."
        ),
        tag: Optional[str] = Field(
            default=None, description="hashtag: the hashtag, without #."
        ),
        region: Optional[str] = Field(
            default=None, description="trending: two-letter country code (default US)."
        ),
        sort: Optional[str] = Field(
            default=None, description="search: sort option, when TikTok supports it."
        ),
        count: Optional[int] = Field(
            default=None,
            description="List actions: maximum items (1-100; default 10 for search, 20 otherwise).",
        ),
        cursor: Optional[str] = Field(
            default=None,
            description=(
                "next_cursor from the previous page of the same action and arguments. "
                "Not for trending."
            ),
        ),
    ) -> Union[dict, str]:
        """
        Read public TikTok data. Pick an action; set only its arguments:
        search (query, count, cursor, sort): videos for keywords.
        post (id): one video.
        profile (username): a profile.
        profile_posts (username, count, cursor): a user's videos.
        comments (id, count, cursor): comments on a video.
        hashtag (tag, count, cursor): videos with a hashtag.
        trending (region, count): trending videos, one page.

        List actions return {"results": [...], "next_cursor": ...}. For more, repeat the call with cursor=next_cursor (null on the last page). post and profile return one object; errors return a string.
        """
        return self._social(
            "tiktok",
            action,
            query=query,
            id=id,
            username=username,
            tag=tag,
            region=region,
            sort=sort,
            count=count,
            cursor=cursor,
        )

    def facebook_tool(
        self,
        action: FacebookAction = Field(
            description="What to read; see the tool description."
        ),
        query: Optional[str] = Field(
            default=None, description="search: keyword, phrase or hashtag."
        ),
        id: Optional[str] = Field(
            default=None, description="post and comments: the post or media ID."
        ),
        profile_id: Optional[str] = Field(
            default=None,
            description="profile and profile_posts: page or profile ID, username or vanity handle.",
        ),
        tag: Optional[str] = Field(
            default=None, description="hashtag: keyword or hashtag, without #."
        ),
        count: Optional[int] = Field(
            default=None,
            description="List actions: maximum items (1-100; default 10 for search, 20 otherwise).",
        ),
        cursor: Optional[str] = Field(
            default=None,
            description="next_cursor from the previous page of the same action and arguments.",
        ),
    ) -> Union[dict, str]:
        """
        Read public Facebook data. Pick an action; set only its arguments:
        search (query, count, cursor): Meta Ad Library ads for keywords (not organic posts).
        post (id): one post.
        profile (profile_id): a page or profile.
        profile_posts (profile_id, count, cursor): a page's posts.
        comments (id, count, cursor): comments on a post (about 10 per page).
        hashtag (tag, count, cursor): Meta Ad Library ads for a hashtag.

        List actions return {"results": [...], "next_cursor": ...}. For more, repeat the call with cursor=next_cursor (null on the last page). post and profile return one object; errors return a string.
        """
        return self._social(
            "facebook",
            action,
            query=query,
            id=id,
            profile_id=profile_id,
            tag=tag,
            count=count,
            cursor=cursor,
        )

    def instagram_tool(
        self,
        action: InstagramAction = Field(
            description="What to read; see the tool description."
        ),
        query: Optional[str] = Field(
            default=None, description="search: keyword, username or phrase."
        ),
        id: Optional[str] = Field(
            default=None,
            description=(
                "post: shortcode, media ID or post URL. comments: the shortcode only "
                "(DeuOR-CjKE9 in instagram.com/p/DeuOR-CjKE9/)."
            ),
        ),
        username: Optional[str] = Field(
            default=None, description="profile and profile_posts: username, without @."
        ),
        tag: Optional[str] = Field(
            default=None, description="hashtag: the hashtag, without #."
        ),
        count: Optional[int] = Field(
            default=None,
            description="List actions: maximum items (1-100; default 10 for search, 20 otherwise).",
        ),
        cursor: Optional[str] = Field(
            default=None,
            description=(
                "next_cursor from the previous page of the same action and arguments. "
                "Not for search."
            ),
        ),
    ) -> Union[dict, str]:
        """
        Read public Instagram data. Pick an action; set only its arguments:
        search (query, count): profiles for keywords, one page.
        post (id): one post.
        profile (username): a user's profile.
        profile_posts (username, count, cursor): a user's posts.
        comments (id, count, cursor): comments on a post.
        hashtag (tag, count, cursor): posts with a hashtag.

        List actions return {"results": [...], "next_cursor": ...}. For more, repeat the call with cursor=next_cursor (null on the last page). post and profile return one object; errors return a string.
        """
        return self._social(
            "instagram",
            action,
            query=query,
            id=id,
            username=username,
            tag=tag,
            count=count,
            cursor=cursor,
        )

    # ─── People Search ───────────────────────────────────────────────────────

    def people_search_tool(
        self,
        query: Optional[str] = Field(
            default=None,
            description=(
                "Who to find, in plain language, e.g. 'Heads of sales at fintech "
                "startups who post about outbound' (3-1000 characters). Required "
                "unless task_id is set."
            ),
        ),
        platform: Optional[PeoplePlatform] = Field(
            default=None,
            description=(
                "x (X accounts), linkedin (LinkedIn members), youtube (YouTube "
                "creators) or tiktok (TikTok creators). Required unless task_id is set."
            ),
        ),
        count: int = Field(
            default=25,
            description=(
                "How many qualified people to find (5-200). The search reserves this "
                "many profiles and bills only the ones it delivers."
            ),
        ),
        min_score: Optional[int] = Field(
            default=None, description="Only return people with at least this fit score (0-100)."
        ),
        exclude: Optional[List[str]] = Field(
            default=None,
            description="Handles to leave out, e.g. people you already contacted.",
        ),
        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description=(
                "Platform filters. x: min_followers, max_followers, verified_only, "
                "countries, keywords, exclude_keywords. linkedin: min_followers, "
                "max_followers, countries, keywords, exclude_keywords. youtube: "
                "min_subscribers, max_subscribers, countries, require_contact, "
                "keywords, exclude_keywords. tiktok: min_followers, max_followers, "
                "require_contact, keywords, exclude_keywords. countries, keywords and "
                "exclude_keywords are lists of strings."
            ),
        ),
        idempotency_key: Optional[str] = Field(
            default=None,
            description=(
                "Any unique string. Sending the same key again returns the first "
                "search instead of starting and billing a new one."
            ),
        ),
        timeout: int = Field(
            default=300,
            description=(
                "Seconds to wait for the search to finish (0-1800). If it is still "
                "running, the people found so far are returned with the task_id."
            ),
        ),
        task_id: Optional[str] = Field(
            default=None,
            description=(
                "task_id of an earlier search. Waits on it and returns its people "
                "instead of starting a new search. Free."
            ),
        ),
    ) -> Union[dict, str]:
        """
        Find people on X, LinkedIn, YouTube or TikTok who match a plain-language description.

        BILLED: reserves `count` profiles at the route price; you pay only for profiles delivered,
        and a search that finds no one is free. SLOW: searches run 1-5 minutes. Waits up to
        `timeout` seconds, then returns the people found so far with the task_id. Call again
        with only task_id to keep waiting (free; starts no new search).

        Returns:
            dict | str: Task status, cost and compact people (name, handle, url, audience,
            location, score, why, contacts, links), or an error string.
        """
        try:
            query, platform, count, min_score = map(
                _arg, (query, platform, count, min_score)
            )
            exclude, filters, idempotency_key = map(
                _arg, (exclude, filters, idempotency_key)
            )
            timeout, task_id = _arg(timeout), _arg(task_id)
            if timeout is None or not 0 <= timeout <= 1800:
                raise ValueError("timeout must be between 0 and 1800 seconds.")
            kwargs: Dict[str, Any] = {
                "tool_name": "people_search_tool",
                "timeout": timeout,
                "poll_interval": self.poll_interval,
            }
            if task_id:
                kwargs["task_id"] = task_id
            else:
                if not query or not platform:
                    raise ValueError(
                        "Set query and platform to start a People Search, or task_id "
                        "to keep waiting on one."
                    )
                if filters is not None and not isinstance(filters, dict):
                    raise ValueError("filters must be an object of platform filters.")
                kwargs["platform"] = platform
                kwargs["idempotency_key"] = idempotency_key
                kwargs["body"] = people_search_body(
                    platform,
                    query,
                    count=count,
                    min_score=min_score,
                    exclude=exclude,
                    **(filters or {}),
                )
            return _run_desearch(self.client, lambda c: search_people(c, **kwargs))
        except Exception as e:
            return self._error(e)

    def people_profiles_tool(
        self,
        platform: PeoplePlatform = Field(
            description="Whose people to list: x, youtube, linkedin or tiktok."
        ),
        q: Optional[str] = Field(
            default=None, description="Text in the name, handle, bio or location."
        ),
        sort: Literal["score", "recent", "audience"] = Field(
            default="score",
            description="score (best fit first), recent or audience (largest first).",
        ),
        limit: int = Field(default=25, description="People per page (1-200)."),
        cursor: Optional[str] = Field(
            default=None, description="next_cursor from the previous page."
        ),
        has: Optional[List[PeopleChannel]] = Field(
            default=None,
            description="Channels every person must have, e.g. ['email'] or ['email', 'instagram'].",
        ),
        min_audience: Optional[int] = Field(
            default=None, description="Minimum followers (subscribers on YouTube)."
        ),
        max_audience: Optional[int] = Field(
            default=None, description="Maximum followers (subscribers on YouTube)."
        ),
        min_score: Optional[int] = Field(
            default=None, description="Minimum fit score (0-100)."
        ),
        location: Optional[str] = Field(
            default=None, description="Text in the location, e.g. 'united states'."
        ),
        verified: Optional[bool] = Field(
            default=None, description="True keeps only verified accounts (x, tiktok)."
        ),
    ) -> Union[dict, str]:
        """
        List the people earlier Desearch People Searches found on one platform, one entry per person.

        Free and instant: reads saved results and never starts a search. Filter by text, channels
        they have (has), audience range, minimum score, location and verified; sort by score,
        recent or audience.

        Returns:
            dict | str: total, compact people (with the search that found them) and next_cursor,
            or an error string.
        """
        try:
            return self._people_profiles(
                **{
                    name: _arg(value)
                    for name, value in {
                        "platform": platform,
                        "q": q,
                        "sort": sort,
                        "limit": limit,
                        "cursor": cursor,
                        "has": has,
                        "min_audience": min_audience,
                        "max_audience": max_audience,
                        "min_score": min_score,
                        "location": location,
                        "verified": verified,
                    }.items()
                }
            )
        except Exception as e:
            return self._error(e)

    def _people_profiles(self, **filters: Any) -> dict:
        return _run_desearch(
            self.client,
            lambda c: list_people_profiles(
                c, tool_name="people_profiles_tool", **filters
            ),
        )
