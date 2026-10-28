from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing import Any, Dict, Optional, List, Literal, Union
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from desearch_py import Desearch

from llama_index_desearch.companies import (
    CompanyClient,
    company_enrich_body,
    company_search_body,
    control_company_task,
    list_company_tasks,
    run_company_task,
)
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
CompanyTaskType = Literal["search", "enrich"]
CompanyTaskAction = Literal["pause", "resume", "cancel"]

_COMPANY_ROLES_DESCRIPTION = (
    "Leaders to find: ceo, founder, cto, cfo, coo, cmo, president, owner, "
    "managing_director. Default ceo, founder, cto, cfo."
)
_COMPANY_INFER_DESCRIPTION = (
    "Guess leaders' emails from the company's address pattern (default true). "
    "Inferred emails say source 'inferred' and have a confidence."
)
_COMPANY_VERIFY_DESCRIPTION = "Check every email: syntax, mail server and web (default true)."
_COMPANY_METADATA_DESCRIPTION = (
    "Your own labels for the task, e.g. {'campaign': 'q4-outbound'}: up to 10 keys of "
    "1-40 characters (a-z, 0-9, '_', '.', '-'), string values up to 200 characters."
)
_COMPANY_IDEMPOTENCY_DESCRIPTION = (
    "Any unique string. Sending the same key again returns the first task instead of "
    "starting and billing a new one."
)
_COMPANY_TIMEOUT_DESCRIPTION = (
    "Seconds to wait for the task to finish (0-1800). If it is still running, the "
    "companies delivered so far are returned with the task_id."
)
_COMPANY_TASK_ID_DESCRIPTION = (
    "task_id of an earlier Company Search or Company Enrich task. Waits on it and "
    "returns its companies instead of starting a new task. Free."
)
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
        "company_search_tool",
        "company_enrich_tool",
        "company_tasks_tool",
        "company_task_control_tool",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""

        self.client = Desearch(api_key=api_key)
        # People Search and Company Search call the API over HTTP until desearch-py
        # 1.4.0 (People Search) and 1.8.0 (Company Search) are on PyPI.
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

    def company_search_tool(
        self,
        query: Optional[str] = Field(
            default=None,
            description=(
                "The companies to find, in plain language: what they do, where, how far "
                "along, e.g. 'Series A fintech startups in Berlin building payment "
                "infrastructure' (3-1000 characters). Required unless task_id is set."
            ),
        ),
        count: int = Field(
            default=25,
            description=(
                "How many companies to deliver (5-200). The task reserves this many and "
                "bills only the companies it delivers."
            ),
        ),
        min_score: Optional[int] = Field(
            default=None,
            description="Only deliver companies with at least this fit score (0-100).",
        ),
        exclude: Optional[List[str]] = Field(
            default=None,
            description="Domains or URLs to leave out, e.g. companies you already contacted (up to 1000).",
        ),
        countries: Optional[List[str]] = Field(
            default=None,
            description="Where the companies should be: ISO codes or names, e.g. ['DE', 'Austria'] (up to 20).",
        ),
        require: Optional[List[str]] = Field(
            default=None,
            description=(
                "Only deliver companies that have all of these: email, phone, linkedin "
                "(the company's LinkedIn page), leader (one of roles), leader_email. A "
                "company without them is not delivered or billed."
            ),
        ),
        roles: Optional[List[str]] = Field(
            default=None, description=_COMPANY_ROLES_DESCRIPTION
        ),
        infer_emails: Optional[bool] = Field(
            default=None, description=_COMPANY_INFER_DESCRIPTION
        ),
        verify_emails: Optional[bool] = Field(
            default=None, description=_COMPANY_VERIFY_DESCRIPTION
        ),
        metadata: Optional[Dict[str, str]] = Field(
            default=None, description=_COMPANY_METADATA_DESCRIPTION
        ),
        idempotency_key: Optional[str] = Field(
            default=None, description=_COMPANY_IDEMPOTENCY_DESCRIPTION
        ),
        timeout: int = Field(default=300, description=_COMPANY_TIMEOUT_DESCRIPTION),
        task_id: Optional[str] = Field(
            default=None, description=_COMPANY_TASK_ID_DESCRIPTION
        ),
    ) -> Union[dict, str]:
        """
        Find companies matching a plain-language description, with public contacts and leaders.

        BILLED: reserves `count` companies; you pay only for those delivered (at least 2 if any),
        and a task that delivers none is free. SLOW: runs for minutes. Waits up to `timeout`
        seconds, then returns the companies so far with the task_id. Call again with only task_id
        to keep waiting (free; starts no new task).

        Returns:
            dict | str: Task status, cost and companies (score, why, emails, phones, socials,
            leaders), or an error string.
        """
        try:
            query, count, min_score, exclude = map(_arg, (query, count, min_score, exclude))
            countries, require, roles = map(_arg, (countries, require, roles))
            infer_emails, verify_emails, metadata = map(
                _arg, (infer_emails, verify_emails, metadata)
            )
            idempotency_key, timeout, task_id = map(_arg, (idempotency_key, timeout, task_id))
            body = None
            if not task_id:
                if not query:
                    raise ValueError(
                        "Set query to start a Company Search, or task_id to keep waiting on one."
                    )
                body = company_search_body(
                    query,
                    count=count,
                    min_score=min_score,
                    exclude=exclude,
                    countries=countries,
                    require=require,
                    roles=roles,
                    infer_emails=infer_emails,
                    verify_emails=verify_emails,
                    metadata=metadata,
                )
            return self._company_task(
                "company_search_tool", "search", body, idempotency_key, timeout, task_id
            )
        except Exception as e:
            return str(e)

    def company_enrich_tool(
        self,
        domains: Optional[List[str]] = Field(
            default=None,
            description=(
                "1-200 company domains or URLs, e.g. ['acme.com', 'https://www.beta.io/about']. "
                "Each is reduced to its host without www.; duplicates are removed. Required "
                "unless task_id is set."
            ),
        ),
        roles: Optional[List[str]] = Field(
            default=None, description=_COMPANY_ROLES_DESCRIPTION
        ),
        infer_emails: Optional[bool] = Field(
            default=None, description=_COMPANY_INFER_DESCRIPTION
        ),
        verify_emails: Optional[bool] = Field(
            default=None, description=_COMPANY_VERIFY_DESCRIPTION
        ),
        metadata: Optional[Dict[str, str]] = Field(
            default=None, description=_COMPANY_METADATA_DESCRIPTION
        ),
        idempotency_key: Optional[str] = Field(
            default=None, description=_COMPANY_IDEMPOTENCY_DESCRIPTION
        ),
        timeout: int = Field(default=300, description=_COMPANY_TIMEOUT_DESCRIPTION),
        task_id: Optional[str] = Field(
            default=None, description=_COMPANY_TASK_ID_DESCRIPTION
        ),
    ) -> Union[dict, str]:
        """
        Collect public contacts and leaders for companies you already know, by domain.

        One company per domain with the same fields as company_search_tool and no fit score
        (score is None). BILLED: reserves one company per domain at the Company Enrich price; you pay
        only for companies delivered (at least 1), and a task that delivers none is free. SLOW: runs
        for minutes. Waits up to `timeout` seconds, then returns the companies delivered so far with
        the task_id. Call again with only task_id to keep waiting (free; starts no new task).

        Returns:
            dict | str: Task status, cost and compact companies, or an error string.
        """
        try:
            domains, roles, infer_emails = map(_arg, (domains, roles, infer_emails))
            verify_emails, metadata = _arg(verify_emails), _arg(metadata)
            idempotency_key, timeout, task_id = map(_arg, (idempotency_key, timeout, task_id))
            body = None
            if not task_id:
                if not domains:
                    raise ValueError(
                        "Set domains to start a Company Enrich, or task_id to keep waiting on one."
                    )
                body = company_enrich_body(
                    domains,
                    roles=roles,
                    infer_emails=infer_emails,
                    verify_emails=verify_emails,
                    metadata=metadata,
                )
            return self._company_task(
                "company_enrich_tool", "enrich", body, idempotency_key, timeout, task_id
            )
        except Exception as e:
            return str(e)

    def company_tasks_tool(
        self,
        type: Optional[CompanyTaskType] = Field(
            default=None,
            description="Only Company Search ('search') or only Company Enrich ('enrich') tasks.",
        ),
        status: Optional[List[str]] = Field(
            default=None,
            description=(
                "Only tasks in these statuses, e.g. ['running', 'paused']: queued, running, "
                "pausing, paused, cancelling, completed, partial, failed, cancelled, expired."
            ),
        ),
        limit: int = Field(default=20, description="Tasks per page (1-100)."),
        cursor: Optional[str] = Field(
            default=None,
            description="next_cursor from the previous page. Keep the same filters when you page.",
        ),
        metadata: Optional[Dict[str, str]] = Field(
            default=None,
            description="Only tasks whose metadata has all of these labels (exact match, up to 3).",
        ),
    ) -> Union[dict, str]:
        """
        List your Company Search and Company Enrich tasks, newest first.

        Each task has its status, counts, cost, why it stopped (completion_reason, coverage_limited),
        warnings and metadata labels. Use it to find a task_id, then read the task's companies with
        company_search_tool or company_enrich_tool and only task_id. Free and instant: it never
        starts a task. People Search tasks are not listed.

        Returns:
            dict | str: tasks and next_cursor, or an error string.
        """
        try:
            filters = {
                "type": _arg(type),
                "status": _arg(status),
                "limit": _arg(limit),
                "cursor": _arg(cursor),
                "metadata": _arg(metadata),
            }
            if not 1 <= filters["limit"] <= 100:
                raise ValueError("limit must be between 1 and 100.")
            with CompanyClient(self.api_key, self.base_url) as client:
                return list_company_tasks(
                    client, tool_name="company_tasks_tool", **filters
                )
        except Exception as e:
            return str(e)

    def company_task_control_tool(
        self,
        task_id: str = Field(
            description="task_id of a Company Search or Company Enrich task."
        ),
        action: CompanyTaskAction = Field(
            description=(
                "pause (a running task finishes its current batch first), resume (no new "
                "charge) or cancel (stops it for good; cannot be undone)."
            )
        ),
    ) -> Union[dict, str]:
        """
        Pause, resume or cancel a Company Search or Company Enrich task and return its status.

        Free: it charges and refunds nothing by itself. Pause keeps the companies delivered so far and
        the reservation; a task paused for more than 7 days is cancelled. Resume continues with no
        new charge. Cancel stops it for good: you pay only for the companies already delivered and
        the rest of the reservation is returned. Cancel cannot be undone.

        Returns:
            dict | str: The task's updated status, or an error string.
        """
        try:
            task_id, action = _arg(task_id), _arg(action)
            if not task_id:
                raise ValueError("task_id is required.")
            with CompanyClient(self.api_key, self.base_url) as client:
                return control_company_task(
                    client,
                    task_id=task_id,
                    action=action,
                    tool_name="company_search_tool",
                    control_tool_name="company_task_control_tool",
                    enrich_tool_name="company_enrich_tool",
                )
        except Exception as e:
            return str(e)

    def _company_task(
        self,
        tool_name: str,
        kind: str,
        body: Optional[Dict[str, Any]],
        idempotency_key: Optional[str],
        timeout: Any,
        task_id: Optional[str],
    ) -> dict:
        if timeout is None or not 0 <= timeout <= 1800:
            raise ValueError("timeout must be between 0 and 1800 seconds.")
        kwargs: Dict[str, Any] = {
            "tool_name": tool_name,
            "control_tool_name": "company_task_control_tool",
            "timeout": timeout,
            "poll_interval": self.poll_interval,
        }
        if task_id:
            kwargs["task_id"] = task_id
        else:
            kwargs.update(kind=kind, body=body, idempotency_key=idempotency_key)
        with CompanyClient(self.api_key, self.base_url) as client:
            return run_company_task(client, **kwargs)
