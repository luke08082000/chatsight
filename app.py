import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType, AgentExecutor, Tool, load_tools
from langchain.agents.agent_toolkits.base import BaseToolkit, BaseTool
from custom_jira_action import JiraAction
from custom_jira_api_wrapper import JiraAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
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





class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "AskHuman"
    description = (
        """You can ask the customer for clarifications when you think you need to know more. 
        The input should be a question for the customer.
        Use this to recommend and ask permission to reboot instance.
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
                    Ask the customer if they would like to have a ticket created for them first. Aways provie the ticket link.
                    For example, to create a low priority task called "Unable to login" with description "I am unable to log in to my AWS console.", you would pass in the following dictionary: 
                    {{"summary": "Unable to login", "description": "Unable to log in to my AWS console.", "issuetype": {{"name": "Submit a request or incident"}}, "priority": {{"name": "Low"}}, "project": {{"key": "SD"}}}}
                    After creating an issue, provide the link of the issue to allow status tracking, Example: https://chatsight-vsd.atlassian.net/browse/SD-27
                    Make sure to include the instance id of the ec2 instance in the description.
                    Include your observation of the log into the description.
                    """
            },
            {
                "mode": "update_status",
                "name": "Update Issue Status",
                "description": """
                    This tool is a wrapper around atlassian-python-api's Jira update_status API, useful when you need to update a status of a Jira issue.
                    The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed into atlassian-python-api's Jira `update_status` function.
                    Update status to "In Progress" when used for self-healing and then update again to "Done" after self-healing or update to "Pending" if further investigation is needed.
                    You can only update issues to "Done" when they are "In Progress" so don't forget to update status to "In Progress" when attempting to self-heal.
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
    ec2 = boto3.client('ec2', region_name="ap-southeast-1")
    
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
    ec2 = boto3.client('ec2', region_name="ap-southeast-1")
    response = ec2.get_console_output(InstanceId=instance_id)

    return response['Output']

#Check EC2 Instance Status
def check_instance_status(instance_ids):
    ec2 = boto3.client('ec2', region_name="ap-southeast-1")
    response = ec2.describe_instance_status(InstanceIds=[instance_ids])
    
    for status in response['InstanceStatuses']:
        return(f"Instance ID: {status['InstanceId']}, Status: {status['InstanceState']['Name']}")
    
#Get CPU util
from datetime import datetime, timedelta

def get_ec2_cpu_utilization(instance_id, duration_minutes=60):
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch', region_name='ap-southeast-1')

    # Set the end time to the current time
    end_time = datetime.utcnow()

    # Set the start time to 'duration_minutes' minutes ago
    start_time = end_time - timedelta(minutes=duration_minutes)

    # Get CPU utilization metric data
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'm1',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'CPUUtilization',
                        'Dimensions': [
                            {
                                'Name': 'InstanceId',
                                'Value': instance_id
                            },
                        ]
                    },
                    'Period': 60,
                    'Stat': 'Average',
                },
            },
        ],
        StartTime=start_time,
        EndTime=end_time,
    )

    # Extract and print CPU utilization data
    if 'MetricDataResults' in response and response['MetricDataResults']:
        datapoints = response['MetricDataResults'][0]['Timestamps']
        values = response['MetricDataResults'][0]['Values']
        
        if datapoints and values:
            for timestamp, value in zip(datapoints, values):
                return(f"Timestamp: {timestamp}, CPU Utilization: {value}%")
        else:
            return("No data available for the specified duration.")
    else:
        return("Error retrieving metric data.")

def get_ec2_disk_io(instance_id, duration_minutes=60):
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch', region_name='ap-southeast-1')

    # Set the end time to the current time
    end_time = datetime.utcnow()

    # Set the start time to 'duration_minutes' minutes ago
    start_time = end_time - timedelta(minutes=duration_minutes)

    # Get disk I/O metrics data
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'm1',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'DiskReadBytes',
                        'Dimensions': [
                            {
                                'Name': 'InstanceId',
                                'Value': instance_id
                            },
                        ]
                    },
                    'Period': 60,
                    'Stat': 'Sum',
                },
            },
            {
                'Id': 'm2',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'DiskWriteBytes',
                        'Dimensions': [
                            {
                                'Name': 'InstanceId',
                                'Value': instance_id
                            },
                        ]
                    },
                    'Period': 60,
                    'Stat': 'Sum',
                },
            },
        ],
        StartTime=start_time,
        EndTime=end_time,
    )

    # Extract and print disk I/O metrics data
    if 'MetricDataResults' in response and response['MetricDataResults']:
        read_bytes = response['MetricDataResults'][0]['Values'][0]
        write_bytes = response['MetricDataResults'][1]['Values'][0]

        return f"Disk Read Bytes: {read_bytes}, Disk Write Bytes: {write_bytes}"
    else:
        return "Error retrieving disk I/O metrics data."
    
def get_ec2_memory_usage(instance_id, duration_minutes=60):
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch', region_name='ap-southeast-1')

    # Set the end time to the current time
    end_time = datetime.utcnow()

    # Set the start time to 'duration_minutes' minutes ago
    start_time = end_time - timedelta(minutes=duration_minutes)

    # Get memory usage metric data
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'm1',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'MemoryUtilization',
                        'Dimensions': [
                            {
                                'Name': 'InstanceId',
                                'Value': instance_id
                            },
                        ]
                    },
                    'Period': 60,
                    'Stat': 'Average',
                },
            },
        ],
        StartTime=start_time,
        EndTime=end_time,
    )

    # Extract and print memory usage metrics data
    if 'MetricDataResults' in response and response['MetricDataResults']:
        metric_data_results = response['MetricDataResults'][0]

        if 'Values' in metric_data_results and metric_data_results['Values']:
            memory_utilization = metric_data_results['Values'][0]
            return f"Memory Utilization: {memory_utilization}%"
        else:
            return "No data available for the specified duration."

    else:
        return "Error retrieving memory usage metrics data."

def get_cloudwatch_alarms(instance_id):
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch', region_name='ap-southeast-1')

    # List alarms for the specified EC2 instance
    response = cloudwatch.describe_alarms_for_metric(
        MetricName='CPUUtilization',  # Replace with the metric you're interested in
        Namespace='AWS/EC2',
        Dimensions=[
            {
                'Name': 'InstanceId',
                'Value': instance_id
            },
        ]
    )

    # Extract and print alarm information
    alarms = response.get('MetricAlarms', [])
    if alarms:
        for alarm in alarms:
            print(f"Alarm Name: {alarm['AlarmName']}")
            print(f"Alarm State: {alarm['StateValue']}")
            print(f"Alarm Actions: {alarm.get('AlarmActions', [])}")
            print("----")
    else:
        print("No alarms found for the specified EC2 instance.")

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
            description="""Only call after getting permission from customer.
            Only call after instance id has been collected.
            Ask customer if they would like to create a ticket first.
            This is one of the self-healing/troubleshooting tools. 
            Suggest reboot when dealing with performance issues.
            This reboots an EC2 instance when run. Useful when customer complains about Instance Performance Issues.
            Pass the instance id as the action input.
            """
        ),
        # Tool(
        #     name="Get EC2 Console Output",
        #     func=get_console_output,
        #     description="""Only call after instance id has been collected.
        #     This is one of the self-healing/troubleshooting tools used to diagnose.
        #     This is used to retrieve console output logs of an EC2 instance.
        #     Useful for diagnosing boot issues, error messages during startup, or other console-related problems.
        #     Pass the instance id as the action input."""
        # ),
        Tool(
            name="Get Disk I/O",
            func=get_ec2_disk_io,
            description="""Only call after instance id has been collected.
            This is one of the self-healing/troubleshooting tools used to diagnose.
            This is used to check the Disk I/O of an EC2 instance. Useful for diagnosing performance issues.
            Pass the instance id as the action input."""
        ),
        # Tool(
        #     name="Get Cloudwatch Alarms",
        #     func=get_cloudwatch_alarms,
        #     description="""Only call after instance id has been collected.
        #     This is one of the self-healing/troubleshooting tools used to diagnose.
        #     This is used to check if alarms have been set off for an EC2 instance's CPU Utilization. Useful for diagnosing performance issues.
        #     Pass the instance id as the action input."""
        # ),
        # Tool(
        #     name="Check EC2 Instance Status",
        #     func=check_instance_status,
        #     description="""Only call after instance id has been collected.
        #     This is one of the self-healing/troubleshooting tools.
        #     This is used to check the status of an EC2 instance.
        #     Pass the instance id as the action input."""
        # ),
        Tool(
            name="Get CPU Utilization",
            func=get_ec2_cpu_utilization,
            description="""Only call after instance id has been collected.
            This is one of the self-healing/troubleshooting tools used to diagnose.
            This is used to check the CPU Utilization of an EC2 instance.
            Pass the instance id as the action input. Make sure it does not contain non-ASCII characters."""
        ),
        HumanInputChainlit()
    ] + jira_toolkit.get_tools()


    memory = ConversationBufferMemory(
        memory_key='history',
        return_messages=True
    )

    PREFIX = """"You are a customer service assistant. Answer questions politely.
    Be transparent. Always say your observations using the troubleshooting tools so that the customer knows what's happening.
    Ask the customer if they would like to have a ticket created for them before creating one.
    Always provide the link of the ticket after creating one and after the customer has provided their instance id.
    Don't put "Thought: " before your actual thought.
    Don't forget to provide the ticket link after asking if you can create a ticket.

    These are the scenarios that can happen: 
    1. The customer asks an FAQ and you will answer using Knowledge Retrieval.
    2. The customer reports an issue that can be solved using one of the self-healing tools.
    In this case, first ask if you can create a ticket (Create Issue), then provide the ticket link then update issue to "In Progress", then use the self-healing tools to diagnose and recommend solution to the isse, then ask the customer if it solved the problem (AskHuman),
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
        max_iterations=25,
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