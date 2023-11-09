import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, AgentExecutor, Tool
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMMathChain
from chainlit.sync import run_sync
import chainlit as cl

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a customer service assistant. 
Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up 
an answer. For more complex issues that require assistance or follow-up, suggest to the user that you can help them create support tickets.
If it's a greeting, just greet back politely.

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

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
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
            description="useful when you need to answer questions about AWS."
        ), HumanInputChainlit()
    ] + jira_toolkit.get_tools()


    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3,
        return_messages=True
    )

    conversational_agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        verbose=True,
        max_iterations=10,
        early_stopping_method='generate',
        memory=memory,
        handle_parsing_errors=True
    )

    cl.user_session.set("agent", conversational_agent)



@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    prompt = """Context: You are a customer service assistant.
        Task: You will be asked questions about AWS and you will answer them. You may also be asked to create an issue, and you will guide me on the details the I need to provide: a short description of the issue and priority level(low, medium, high).
        Short description and priority level must always be provided before creation of issue if not already provided. Ask customer first using HumanInputChain tool before creating an issue.
        For create issue, always create issues from project Service Desk with the project key of SD. The issue type is 'Submit a request or incident'. Project key is "SD".
        Make sure to include Project key "SD" when creating an issue.
        Input:"""
    res = await agent.run(prompt +
        message
    )
    await cl.Message(content=res).send()