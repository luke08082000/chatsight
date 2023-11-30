from typing import Optional


from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from custom_jira_api_wrapper import JiraAPIWrapper


class JiraAction(BaseTool):
    """Tool that queries the Atlassian Jira API."""

    api_wrapper: JiraAPIWrapper
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Atlassian Jira API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)