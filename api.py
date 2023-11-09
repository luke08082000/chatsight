import os
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool, AgentExecutor, OpenAIMultiFunctionsAgent, AgentOutputParser
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.llms import OpenAI
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.utilities import SerpAPIWrapper
from typing import List, Union
from langchain.chat_models import ChatOpenAI
from langchain.utilities.jira import JiraAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain import PromptTemplate, OpenAI, LLMChain
import requests
import re

from dotenv import load_dotenv

load_dotenv()

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-4-0613'
)
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
Question: {input}
{agent_scratchpad}"""


chain = LLMChain(
    prompt = PromptTemplate.from_template(template),
    llm = llm
)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

try:

    llm = OpenAI(temperature=0)
    jira = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira)
    agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run("""Context: You are a customer service assistant. When using JIRA toolkit: For issue creation, issues are always created in project Service Desk with the project key of SD. 
    For issue GET request, always get issues from project Service Desk with the project key of SD. The issue type is Submit a request or incident.
    Input: create issue: I am unable to reset my AWS password.
    """)

except requests.exceptions.HTTPError as e:
    error_code = e.response.status_code
    error_message = e.response.text
    print(error_code)
    print(error_message)
