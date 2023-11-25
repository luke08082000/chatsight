import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, AgentExecutor, Tool, load_tools
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.agents.agent_toolkits.base import BaseToolkit, BaseTool
from langchain.utilities.jira import JiraAPIWrapper
from langchain.tools.jira.tool import JiraAction
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMMathChain
from chainlit.sync import run_sync
import chainlit as cl
import boto3

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

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "AskHuman"
    description = (
        """You can ask the customer for clarifications when you think you need to know more. 
        The input should be a question for the customer.
        Use this to ask customer's instance id"""
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query, timeout=300).send())
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
                    Include your observation of the log into the description.
                    """
            },
                        {
                "mode": "update_status",
                "name": "Update Issue Status",
                "description": """
                    This tool is a wrapper around atlassian-python-api's Jira update_status API, useful when you need to update a status of a Jira issue.
                    The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed into atlassian-python-api's Jira `update_status` function.
                    Update status to "In Progress" when used for self-healing and then update again to "Done" after self-healing.
                    For example, to update an issue status of an issue with an id of "SD-19" to "In Progress" you would pass in the following: 
                    {{"issue_key": "SD-19", "status": "In Progress"}}
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

#Self-healing - reboot ec2 instance 
import time

def reboot_ec2_instance(instance_id):
    ec2 = boto3.client('ec2', region_name="ap-southeast-2")
    
    # Initiating the reboot
    ec2.reboot_instances(InstanceIds=[instance_id])
    
    # Wait for a short duration to allow the reboot to take effect
    time.sleep(5)
    
    # Check the instance state after the reboot
    instance = ec2.describe_instances(InstanceIds=[instance_id])
    
    # Extract the state information
    state = instance['Reservations'][0]['Instances'][0]['State']['Name']
    
    # Validate if the state is 'running,' indicating a successful reboot
    if state == 'running':
        return f"EC2 instance {instance_id} successfully rebooted."
    else:
        return f"Failed to reboot EC2 instance {instance_id}. Current state: {state}"
    
#Retrieve EC2 Instance Console Output
def get_console_output(instance_id):
    ec2 = boto3.client('ec2')
    response = ec2.get_console_output(InstanceId=instance_id)

    return response['Output']

#Check EC2 Instance Status
def check_instance_status(instance_ids):
    ec2 = boto3.client('ec2')
    response = ec2.describe_instance_status(InstanceIds=[instance_ids])
    
    for status in response['InstanceStatuses']:
        return(f"Instance ID: {status['InstanceId']}, Status: {status['InstanceState']['Name']}")

##Chainlit code
@cl.on_chat_start
def start():
    jira = JiraAPIWrapper()
    jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira)


    tools = [
        Tool(
            name="Knowledge Retrieval",
            func=qa_bot(),
            description="""Useful when you need to answer customer queries and troubleshoot issues."""
        ),
        Tool(
            name="Reboot EC2 Instance",
            func=reboot_ec2_instance,
            description="""Only call after instance id has been collected.
            This is one of the self-healing/troubleshooting tools. 
            This reboots an EC2 instance when run. Useful when customer complains about Instance Performance Issues.
            Pass the instance id as the action input.
            """
        ),
        Tool(
            name="Get EC2 Console Output",
            func=get_console_output,
            description="""Only call after instance id has been collected.
            This is one of the self-healing/troubleshooting tools.
            This is used to retrieve console output logs of an EC2 instance.
            Useful for diagnosing boot issues, error messages during startup, or other console-related problems.
            Pass the instance id as the action input."""
        ),
        Tool(
            name="Check EC2 Instance Status",
            func=check_instance_status,
            description="""Only call after instance id has been collected.
            This is one of the self-healing/troubleshooting tools.
            This is used to check the status of an EC2 instance.
            Pass the instance id as the action input."""
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

    There are the scenarios that can happen: 
    1. The customer asks an FAQ and you will answer using Knowledge Retrieval.
    2. The customer reports an issue, you will ask the details of the issue (AskHuman) and create an issue (Create Issue).
    3. The customer reports an issue that can be solved using one of the self-healing tools (Reboot EC2 Instance, Get EC2 Console Output, Check EC2 Instance Status).
    In this case, first create a ticket (Create Issue), then run the self-healing tool, then ask the customer if it solved the problem (AskHuman),
    then lastly, Update Issue Status to "Done" if problem was solved, if not Update Issue Status to "Pending".
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

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
        verbose=True,
        max_iterations=8,
        early_stopping_method='generate',
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix':PREFIX,
            'format_instructions':FORMAT_INSTRUCTIONS,
            'suffix':SUFFIX
    }
    )
    cl.user_session.set("agent", agent)



@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()