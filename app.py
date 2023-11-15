import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, AgentExecutor, Tool
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.agents.agent_toolkits.base import BaseToolkit, BaseTool
from langchain.utilities.jira import JiraAPIWrapper
from langchain.tools.jira.tool import JiraAction
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMMathChain
from chainlit.sync import run_sync
import chainlit as cl

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a customer service assistant. 
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know the answer, don't try to make up 
an answer. For more complex issues that require assistance or follow-up, suggest to the user that you can help them create a support ticket.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt


#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', 
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

#Loading the model
def load_llm():
    # llm = OpenAI(temperature=0)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo'
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo'
    )
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa



jira = JiraAPIWrapper()
jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)
search = DuckDuckGoSearchRun()
llm_math_chain = LLMMathChain.from_llm(llm=load_llm(), verbose=True)

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "AskHuman"
    description = (
        "You can ask the customer for clarifications when you think you "
        "need to know more. "
        "The input should be a question for the customer."
        "Only use when you need clarification"
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["content"]
    

class JiraToolkit(BaseToolkit):
    """Jira Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_jira_api_wrapper(cls, jira_api_wrapper: JiraAPIWrapper) -> "JiraToolkit":
        operations: List[Dict] = [
            {
                "mode": "create_issue",
                "name": "Create Issue",
                "description": """
                    This tool is a wrapper around atlassian-python-api's Jira issue_create API, useful when you need to create a Jira issue.
                    The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed into atlassian-python-api's Jira `issue_create` function.
                    For example, to create a low priority task called "Unable to login" with description "I am unable to log in to my AWS console.", you would pass in the following dictionary: 
                    {{"summary": "Unable to login", "description": "Unable to log in to my AWS console.", "issuetype": {{"name": "Submit a request or incident"}}, "priority": {{"name": "Low"}}, "project": {{"key": "SD"}}}}
                    After creating an issue, provide the link of the issue to allow status tracking, Example: https://chatsight-vsd.atlassian.net/browse/SD-27
                    """
            }
        ]
        tools = [
            JiraAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=jira_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools

##Chainlit code
@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    jira = JiraAPIWrapper()
    jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)


    tools = [
        Tool(
            name="Knowledge Retrieval",
            func=qa_bot(),
            description="useful when you need to answer customer queries."
        ), 
        HumanInputChainlit()
    ] + jira_toolkit.get_tools()


    memory = ConversationBufferMemory(
        memory_key='history',
        return_messages=True
    )


    PREFIX = """"You are a customer service assistant. Answer questions politely.
    Look for solutions first using Knowledge Retrieval.
    After answering a question, ask the customer if they found the response helpful. 
    If you've used Knowledge Retrieval and still don't know the answer, ask customer if they want to create a support ticket.
    If you're going to create a ticket, ask customer about the details of the issue and possible error messages.
    Use knowledge retrieval first to check if there is a solution before thinking about creating a ticket.
    """

    FORMAT_INSTRUCTIONS = """Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question"""
    SUFFIX = """
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    conversational_agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
        verbose=True,
        max_iterations=4,
        early_stopping_method='generate',
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix':PREFIX,
            'format_instructions':FORMAT_INSTRUCTIONS,
            'suffix':SUFFIX
    }
    )
    cl.user_session.set("agent", conversational_agent)



@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()