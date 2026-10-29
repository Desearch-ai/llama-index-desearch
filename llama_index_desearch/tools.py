from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing import Any, Dict, Optional, List, Literal, Union
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from desearch_py import Desearch

from llama_index_desearch.people import (
    PEOPLE_API_BASE_URL,
    PeopleClient,
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


def _arg(value: Any) -> Any:
    """Use a Field's default when a spec function is called directly in Python."""
    if isinstance(value, FieldInfo):
        default = value.get_default(call_default_factory=True)
        return None if default is PydanticUndefined else default
    return value


class DesearchToolSpec(BaseToolSpec):
    """Desearch tool spec."""

    spec_functions = [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
        "people_search_tool",
        "people_profiles_tool",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""

        self.client = Desearch(api_key=api_key)
        # People Search calls the API over HTTP until desearch-py 1.4.0 is on PyPI.
        self.api_key = api_key
        self.base_url = PEOPLE_API_BASE_URL
        self.poll_interval = 5.0

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
            description="The model to use for the search. Value should 'NOVA', 'ORBIT' or 'HORIZON'",
        ),
        date_filter: Optional[str] = Field(
            default=None, description="Date filter for the search."
        ),
    ) -> str | dict:
        """
        Perform a search using Desearch.

        Args:
            prompt (str): The search prompt or query.
            tool (List[Literal["web", "hackernews", "reddit", "wikipedia", "youtube", "twitter", "arxiv"]]): List of tools to use. Must include at least one tool.
            model (str, optional): The model to use for the search. Defaults to "NOVA".
            date_filter (Optional[str], optional): Date filter for the search. Defaults to None.

        Returns:
            str | dict: The search result or an error string.
        """
        try:
            result = self.client.search(
                prompt,
                tool,
                model,
                date_filter,
            )

            return result
        except Exception as e:
            return str(e)

    def twitter_search_tool(
        self,
        query: str = Field(description="The Twitter search query."),
        sort: str = Field(default="Top", description="Sort order for the results."),
        count: int = Field(default=10, description="Number of results to return."),
    ) -> BasicTwitterSearchResponse:
        """
        Perform a basic Twitter search using the Exa API.

        Args:
            query (str, optional): The Twitter search query. Defaults to None.
            sort (str, optional): Sort order for the results. Defaults to "Top".
            count (int, optional): Number of results to return. Defaults to 10.

        Returns:
            BasicTwitterSearchResponse: The search results.

        Raises:
            Exception: If an error occurs when calling the API.
        """
        try:
            result = self.client.basic_twitter_search(query, sort, count)
            return result
        except Exception as e:
            return str(e)

    def web_search_tool(
        self,
        query: str = Field(description="The search query."),
        num: int = Field(default=10, description="Number of results to return."),
        start: int = Field(
            default=1, description="The starting index for the search results."
        ),
    ) -> List[WebSearchResult]:
        """
        Perform a basic web search using the Exa API.

        Args:
            query (str, optional): The search query. Defaults to None.
            num (int, optional): Number of results to return. Defaults to 10.
            start (int, optional): The starting index for the search results. Defaults to 1.

        Returns:
            List[WebSearchResult]: The search results.

        Raises:
            Exception: If an error occurs when calling the API.
        """
        try:
            result = self.client.basic_web_search(query, num, start)
            return result
        except Exception as e:
            return str(e)

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
            with PeopleClient(self.api_key, self.base_url) as client:
                return search_people(client, **kwargs)
        except Exception as e:
            return str(e)

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
            return str(e)

    def _people_profiles(self, **filters: Any) -> dict:
        with PeopleClient(self.api_key, self.base_url) as client:
            return list_people_profiles(
                client, tool_name="people_profiles_tool", **filters
            )
