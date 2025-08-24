#!/usr/bin/env python3
"""
Fetch Jira issues with status Done and determine who marked them Done
by inspecting the issue changelog.

Authentication options:
- Cookie header: provide AWSALB and JSESSIONID
- Basic auth: email/username + token/password (optional fallback)

Configuration can be provided via CLI flags or environment variables:
- JIRA_BASE_URL (e.g., https://your-domain.atlassian.net)
- Cookie auth: JIRA_AWSALB and JIRA_JSESSIONID
- OR basic auth: JIRA_EMAIL (or JIRA_USERNAME) and JIRA_API_TOKEN

Examples:
  export JIRA_BASE_URL="https://your-domain.example.com"
  export JIRA_AWSALB="<awsalb_cookie_value>"
  export JIRA_JSESSIONID="<jsessionid_cookie_value>"
  python jira_done_markers.py --jql "status = Done AND project = ABC ORDER BY updated DESC" --output csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timedelta, date

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_RESULTS = 1000
DEFAULT_CONCURRENCY = 4
DEFAULT_STATUS_NAME = "Done"


@dataclass
class JiraAuth:
    base_url: str
    username_or_email: Optional[str] = None
    api_token_or_password: Optional[str] = None
    awsalb_cookie: Optional[str] = None
    jsessionid_cookie: Optional[str] = None


class JiraClient:
    def __init__(self, auth: JiraAuth, timeout: int = 30) -> None:
        self.base_url = auth.base_url.rstrip("/")
        self.timeout = timeout

        session = requests.Session()
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        # Prefer cookie-based auth if provided; otherwise fall back to basic auth
        if auth.awsalb_cookie and auth.jsessionid_cookie:
            cookie_header = f"AWSALB={auth.awsalb_cookie}; JSESSIONID={auth.jsessionid_cookie}"
            session.headers["Cookie"] = cookie_header
        elif auth.username_or_email and auth.api_token_or_password:
            session.auth = (auth.username_or_email, auth.api_token_or_password)
        else:
            raise ValueError("Missing authentication: provide AWSALB+JSESSIONID cookies or username/token")

        # Robust retry policy including 429
        retry = Retry(
            total=8,
            backoff_factor=1.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self.session = session

    def _get(self, path: str, params: Optional[dict] = None) -> Response:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        # Handle explicit 429 when retries are exhausted
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp

    def search_issues(
        self,
        jql: str,
        fields: Optional[List[str]] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[dict]:
        fields_param = ",".join(fields) if fields else "key"
        issues: List[dict] = []
        start_at = 0
        while start_at < max_results:
            remaining = max_results - start_at
            batch_size = min(page_size, remaining)
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": fields_param,
            }
            try:
                resp = self._get("/rest/api/3/search", params=params)
            except requests.HTTPError as e:  # type: ignore[attr-defined]
                # Fallback for Jira Server/DC which may use API v2
                if getattr(e, "response", None) is not None and e.response.status_code == 404:
                    resp = self._get("/rest/api/2/search", params=params)
                else:
                    raise
            data = resp.json()
            batch = data.get("issues", [])
            issues.extend(batch)
            if start_at + len(batch) >= data.get("total", 0) or not batch:
                break
            start_at += len(batch)
        return issues

    def iter_issue_changelog(self, issue_id: str, page_size: int = 100) -> Iterable[dict]:
        # Fetch changelog via issue expand instead of the /changelog endpoint
        params = {"expand": "changelog"}
        try:
            resp = self._get(f"/rest/api/3/issue/{issue_id}", params=params)
        except requests.HTTPError as e:  # type: ignore[attr-defined]
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                resp = self._get(f"/rest/api/2/issue/{issue_id}", params=params)
            else:
                raise
        data = resp.json()
        changelog = data.get("changelog") or {}
        histories = changelog.get("histories", [])
        for history in histories:
            yield history


def find_last_transition_to_status(
    histories: Iterable[dict], target_status: str
) -> Optional[Tuple[str, str, str]]:
    """
    Return tuple (author_display_name, author_account_id, created_at)
    for the MOST RECENT transition where status changed to target_status.
    If none found, return None.
    """
    target_lower = target_status.lower()
    best: Optional[Tuple[str, str, str]] = None
    best_created: Optional[str] = None

    for history in histories:
        items = history.get("items", [])
        for item in items:
            to_string = (item.get("toString") or "").lower()
            if to_string == target_lower:
                created = history.get("created") or ""
                author = history.get("author", {}) or {}
                display_name = author.get("displayName") or author.get("name") or ""
                account_id = author.get("accountId") or author.get("key") or ""
                if best_created is None or created > best_created:
                    best_created = created
                    best = (display_name, account_id, created)

    return best


def collect_done_markers(
    client: JiraClient,
    issues: List[dict],
    status_name: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> List[dict]:
    def worker(issue: dict) -> dict:
        key = issue.get("key")
        issue_id = issue.get("id") or ""
        summary = (((issue.get("fields") or {}).get("summary")) if issue.get("fields") else None) or ""
        try:
            # Jira changelog endpoint requires issue id; fall back to key if id is missing
            histories = list(client.iter_issue_changelog(issue_id or key))
            result = find_last_transition_to_status(histories, status_name)
            if result:
                display_name, account_id, created = result
            else:
                display_name, account_id, created = "", "", ""
        except Exception as e:  # noqa: BLE001
            display_name, account_id, created = "", "", ""
            print(f"Error processing {key}: {e}", file=sys.stderr)
        return {
            "issue_key": key,
            "summary": summary,
            "done_by_display_name": display_name,
            "done_by_account_id": account_id,
            "done_at": created,
        }

    results: List[dict] = []
    if concurrency and concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, issue) for issue in issues]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
    else:
        for issue in issues:
            results.append(worker(issue))
    # Sort by done_at if available, else by issue_key
    results.sort(key=lambda r: (r.get("done_at") or "", r.get("issue_key") or ""))
    return results


def output_results(rows: List[dict], fmt: str) -> None:
    if fmt == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if fmt == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "issue_key",
                "summary",
                "done_by_display_name",
                "done_by_account_id",
                "done_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return
    # pretty
    for row in rows:
        print(
            f"{row['issue_key']:<15} | {row['done_at']:<24} | {row['done_by_display_name']:<30} | {row['summary']}"
        )


def parse_jira_datetime(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:  # noqa: BLE001
            continue
    return None


def aggregate_daily_counts(rows: List[dict], date_format: str) -> Tuple[List[dict], List[str]]:
    # Collect unique users and per-date counts
    user_set: set[str] = set()
    dated_rows: List[Tuple[date, str]] = []
    for row in rows:
        done_at = row.get("done_at") or ""
        dt = parse_jira_datetime(done_at)
        if not dt:
            continue
        d = dt.date()
        user = row.get("done_by_display_name") or ""
        user_set.add(user)
        dated_rows.append((d, user))

    users = sorted(user_set)
    if not dated_rows:
        return [], users

    min_day = min(d for d, _ in dated_rows)
    max_day = max(d for d, _ in dated_rows)

    # Initialize zeroed grid
    day = min_day
    counts_by_day: Dict[date, Dict[str, int]] = {}
    while day <= max_day:
        counts_by_day[day] = {u: 0 for u in users}
        day += timedelta(days=1)

    # Fill counts
    for d, user in dated_rows:
        if user in counts_by_day.get(d, {}):
            counts_by_day[d][user] += 1

    # Build output rows with formatted date and stable user columns
    output: List[dict] = []
    for d in sorted(counts_by_day.keys()):
        row = {"date": d.strftime(date_format)}
        for u in users:
            row[u] = counts_by_day[d].get(u, 0)
        output.append(row)
    return output, users


def output_daily_counts(count_rows: List[dict], users: List[str], fmt: str) -> None:
    if fmt == "json":
        json.dump(count_rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if fmt == "csv":
        fieldnames = ["date"] + users
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in count_rows:
            writer.writerow(row)
        return
    # pretty
    header = ["date"] + users
    print(" | ".join(f"{h:<20}" for h in header))
    for row in count_rows:
        values = [row["date"]] + [str(row[u]) for u in users]
        print(" | ".join(f"{v:<20}" for v in values))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find who marked Jira issues Done using changelog")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JIRA_BASE_URL", ""),
        help="Jira base URL, e.g., https://your-domain.atlassian.net (env: JIRA_BASE_URL)",
    )
    # Cookie authentication
    parser.add_argument(
        "--awsalb",
        default=os.environ.get("JIRA_AWSALB", ""),
        help="AWSALB cookie value (env: JIRA_AWSALB)",
    )
    parser.add_argument(
        "--jsessionid",
        default=os.environ.get("JIRA_JSESSIONID", ""),
        help="JSESSIONID cookie value (env: JIRA_JSESSIONID)",
    )
    # Basic authentication (fallback)
    parser.add_argument(
        "--username",
        default=os.environ.get("JIRA_USERNAME", os.environ.get("JIRA_EMAIL", "")),
        help="Jira username/email (env: JIRA_USERNAME or JIRA_EMAIL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("JIRA_API_TOKEN", ""),
        help="Jira API token/password (env: JIRA_API_TOKEN)",
    )
    parser.add_argument(
        "--jql",
        default="status = Done",
        help="JQL to select issues (default: 'status = Done')",
    )
    parser.add_argument(
        "--status-name",
        default=DEFAULT_STATUS_NAME,
        help="Status name to detect in changelog (default: Done)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Maximum number of issues to fetch (default: 1000)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Page size for search pagination (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of parallel requests for changelogs (default: 4)",
    )
    parser.add_argument(
        "--output",
        choices=["csv", "json", "pretty"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["none", "daily"],
        default="none",
        help="Aggregate results: none (default) or daily per-user counts",
    )
    parser.add_argument(
        "--date-format",
        default="%b-%d",
        help="Date format for aggregated output (default: %b-%d, e.g., Aug-23)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.base_url:
        print("Missing --base-url or JIRA_BASE_URL", file=sys.stderr)
        return 2

    have_cookies = bool(args.awsalb and args.jsessionid)
    have_basic = bool(args.username and args.token)
    if not (have_cookies or have_basic):
        print(
            "Missing authentication. Provide --awsalb and --jsessionid (preferred) or --username and --token.",
            file=sys.stderr,
        )
        return 2

    auth = JiraAuth(
        base_url=args.base_url,
        username_or_email=(args.username or None),
        api_token_or_password=(args.token or None),
        awsalb_cookie=(args.awsalb or None),
        jsessionid_cookie=(args.jsessionid or None),
    )
    client = JiraClient(auth)

    # Fetch issues with only key and summary to reduce payload
    print("Searching issues...", file=sys.stderr)
    issues = client.search_issues(args.jql, fields=["key", "summary"], page_size=args.page_size, max_results=args.max_results)
    print(f"Found {len(issues)} issues. Fetching changelogs...", file=sys.stderr)

    rows = collect_done_markers(client, issues, status_name=args.status_name, concurrency=args.concurrency)
    if args.aggregate == "daily":
        count_rows, users = aggregate_daily_counts(rows, args.date_format)
        output_daily_counts(count_rows, users, args.output)
    else:
        output_results(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


