import asyncio
import concurrent.futures

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing import Any, Optional, List, Literal
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from desearch_py import Desearch


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


def _call_desearch(client: Desearch, method: str, **kwargs: Any) -> Any:
    """Run an async desearch-py method from a sync spec function.

    Each call opens its own client with the key and base URL of ``client``, so
    its aiohttp session never outlives the event loop it ran on.
    """

    async def call() -> Any:
        request_client = Desearch(api_key=client.api_key, base_url=client.base_url)
        try:
            return await getattr(request_client, method)(**kwargs)
        finally:
            await request_client.close()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _to_plain(asyncio.run(call()))
    # Called from inside a running event loop (for example a notebook cell):
    # run the request on a worker thread's own loop instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return _to_plain(executor.submit(asyncio.run, call()).result())


class DesearchToolSpec(BaseToolSpec):
    """Desearch tool spec."""

    spec_functions = [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""

        self.client = Desearch(api_key=api_key)

    def ai_search_tool(
        self,
        prompt: str = Field(description="The search prompt or query."),
        tool: List[
            Literal[
                "web",
                "hackernews",
                "reddit",
                "wikipedia",
                "youtube",
                "twitter",
                "arxiv",
            ]
        ] = Field(description="List of tools to use. Must include at least one tool."),
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
        Search the web, Hacker News, Reddit, Wikipedia, YouTube, X (Twitter) and arXiv with Desearch AI search.

        Args:
            prompt (str): The search prompt or query.
            tool (List[str]): Sources to search. Must include at least one.
            model (str, optional): Kept for compatibility. The current Desearch API ignores it.
            date_filter (Optional[str], optional): Date window such as "PAST_WEEK". Defaults to None (no filter).

        Returns:
            str | dict: The results per source and a summary, or an error string.
        """
        try:
            prompt, tool, date_filter = map(_arg, (prompt, tool, date_filter))
            if not tool:
                raise ValueError("tool must include at least one source.")
            # desearch-py has no model argument. date_filter is passed even when
            # it is None, because desearch-py before 1.3 defaults it to PAST_24_HOURS.
            return _call_desearch(
                self.client,
                "ai_search",
                prompt=prompt,
                tools=tool,
                date_filter=date_filter,
            )
        except Exception as e:
            return str(e)

    def twitter_search_tool(
        self,
        query: str = Field(description="The X (Twitter) search query."),
        sort: str = Field(default="Top", description="Sort order: 'Top' or 'Latest'."),
        count: int = Field(
            default=10, description="Number of posts to return (1-100)."
        ),
    ) -> str | List[dict]:
        """
        Search posts on X (Twitter) with Desearch.

        Args:
            query (str): The search query.
            sort (str, optional): "Top" or "Latest". Defaults to "Top".
            count (int, optional): Number of posts to return (1-100). Defaults to 10.

        Returns:
            str | List[dict]: Posts with their author and engagement counts, or an error string.
        """
        try:
            query, sort, count = map(_arg, (query, sort, count))
            return _call_desearch(
                self.client, "x_search", query=query, sort=sort, count=count
            )
        except Exception as e:
            return str(e)

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
            return str(e)
