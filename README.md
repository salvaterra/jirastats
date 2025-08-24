## Jira Done Markers

Find who marked Jira issues as Done by inspecting the changelog history.

### Setup

1. Create a virtual environment (optional but recommended):
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables (Jira Cloud example):
```bash
export JIRA_BASE_URL="https://your-domain.atlassian.net"
export JIRA_EMAIL="you@example.com"
export JIRA_API_TOKEN="<token>"
```

For Jira Server/Data Center, set `JIRA_USERNAME` and `JIRA_API_TOKEN` (password or PAT).

### Usage

Basic: all Done issues
```bash
python jira_done_markers.py --jql "status = Done" --output csv
```

Filter by project and order by update date:
```bash
python jira_done_markers.py --jql "project = ABC AND status = Done ORDER BY updated DESC" --output pretty
```

Output formats: `csv` (default), `json`, `pretty`.

The script attempts Jira API v3 first and falls back to v2 if needed.

