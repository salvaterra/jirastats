#!/usr/bin/env python3
"""
Fetch Jira issues with status Done and determine who marked them Done
by inspecting the issue changelog.

Authentication:
- Jira Cloud: use email + API token
- Jira Server/DC: use username + password or personal access token as password

Configuration can be provided via CLI flags or environment variables:
- JIRA_BASE_URL (e.g., https://your-domain.atlassian.net)
- JIRA_EMAIL (for Jira Cloud) or JIRA_USERNAME
- JIRA_API_TOKEN (or password)

Example:
  export JIRA_BASE_URL="https://your-domain.atlassian.net"
  export JIRA_EMAIL="you@example.com"
  export JIRA_API_TOKEN="<token>"
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
    username_or_email: str
    api_token_or_password: str


class JiraClient:
    def __init__(self, auth: JiraAuth, timeout: int = 30) -> None:
        self.base_url = auth.base_url.rstrip("/")
        self.timeout = timeout

        session = requests.Session()
        session.auth = (auth.username_or_email, auth.api_token_or_password)
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

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

    def iter_issue_changelog(self, issue_key: str, page_size: int = 100) -> Iterable[dict]:
        start_at = 0
        while True:
            params = {"startAt": start_at, "maxResults": page_size}
            try:
                resp = self._get(f"/rest/api/3/issue/{issue_key}/changelog", params=params)
            except requests.HTTPError as e:  # type: ignore[attr-defined]
                if getattr(e, "response", None) is not None and e.response.status_code == 404:
                    resp = self._get(f"/rest/api/2/issue/{issue_key}/changelog", params=params)
                else:
                    raise
            data = resp.json()
            values = data.get("values", [])
            if not values:
                break
            for history in values:
                yield history
            start_at += len(values)
            if start_at >= data.get("total", 0):
                break


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
            if item.get("field") == "status":
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
        summary = (((issue.get("fields") or {}).get("summary")) if issue.get("fields") else None) or ""
        try:
            histories = list(client.iter_issue_changelog(key))
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find who marked Jira issues Done using changelog")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JIRA_BASE_URL", ""),
        help="Jira base URL, e.g., https://your-domain.atlassian.net (env: JIRA_BASE_URL)",
    )
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.base_url or not args.username or not args.token:
        print(
            "Missing Jira configuration. Provide --base-url, --username, --token or set env vars JIRA_BASE_URL, JIRA_EMAIL/JIRA_USERNAME, JIRA_API_TOKEN",
            file=sys.stderr,
        )
        return 2

    auth = JiraAuth(
        base_url=args.base_url,
        username_or_email=args.username,
        api_token_or_password=args.token,
    )
    client = JiraClient(auth)

    # Fetch issues with only key and summary to reduce payload
    print("Searching issues...", file=sys.stderr)
    issues = client.search_issues(args.jql, fields=["key", "summary"], page_size=args.page_size, max_results=args.max_results)
    print(f"Found {len(issues)} issues. Fetching changelogs...", file=sys.stderr)

    rows = collect_done_markers(client, issues, status_name=args.status_name, concurrency=args.concurrency)
    output_results(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


