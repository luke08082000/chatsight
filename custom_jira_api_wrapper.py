"""Util that calls Jira."""
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

# TODO: think about error handling, more specific api specs, and jql/project limits
class JiraAPIWrapper(BaseModel):
    """Wrapper for Jira API."""

    jira: Any  #: :meta private:
    confluence: Any
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_instance_url: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        jira_username = get_from_dict_or_env(values, "jira_username", "JIRA_USERNAME")
        values["jira_username"] = jira_username

        jira_api_token = get_from_dict_or_env(
            values, "jira_api_token", "JIRA_API_TOKEN"
        )
        values["jira_api_token"] = jira_api_token

        jira_instance_url = get_from_dict_or_env(
            values, "jira_instance_url", "JIRA_INSTANCE_URL"
        )
        values["jira_instance_url"] = jira_instance_url

        try:
            from atlassian import Confluence, Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api is not installed. "
                "Please install it with `pip install atlassian-python-api`"
            )

        jira = Jira(
            url=jira_instance_url,
            username=jira_username,
            password=jira_api_token,
            cloud=True,
        )

        confluence = Confluence(
            url=jira_instance_url,
            username=jira_username,
            password=jira_api_token,
            cloud=True,
        )

        values["jira"] = jira
        values["confluence"] = confluence

        return values


    def issue_create(self, query: str) -> str:
        try:
            import json
        except ImportError:
            raise ImportError(
                "json is not installed. Please install it with `pip install json`"
            )
        params = json.loads(query)
        return self.jira.issue_create(fields=dict(params))


    def update_issue_status(self, issue_key: str, status: str) -> str:
        try:
            # Use the set_issue_status method to transition the issue
            self.jira.set_issue_status(issue_key, status)
            return f"Updated status of issue {issue_key} to {status}"
        except Exception as e:
            return f"Failed to update status. Error: {str(e)}"


    def run(self, mode: str, query: str) -> str:
        if mode == "create_issue":
            return self.issue_create(query)
        elif mode == "update_status":
            import json
            params = json.loads(query)
            print(params["status"])
            return self.update_issue_status(params["issue_key"], params["status"])
        else:
            raise ValueError(f"Got unexpected mode {mode} hahah")


            
