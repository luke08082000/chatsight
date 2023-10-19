import os
from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
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
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo'
    )
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful when you need to answer questions about current events."
    ),
    Tool(
        name="knowledge_retrieval",
        func=qa_bot(),
        description="useful when you need to answer questions about issues in AWS."
    )
]


# agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)
# Creating an agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=load_llm(),
    verbose=True,
    max_iterations=2,
    early_stopping_method='generate',
    memory=memory
)

#Chainlit Code
@cl.on_chat_start
async def start():
    # chain = conversational_agent("I can't login in my AWS account can you help?")
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I'm the InfoAlchemy Bot. What can I help you with?"
    await msg.update()

    cl.user_session.set("chain", memory)

@cl.on_message
async def main(message: cl.Message):
    chain = conversational_agent(message)
    await send_response(chain['output'])
    return chain['output']

async def send_response(response):
    await cl.Message(content=response).send()
