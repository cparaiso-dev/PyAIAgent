import uuid
from functools import partial

from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
#from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch

from tools import chat, get_date
from langchain_tools import parse_langchain_agent_response

load_dotenv()

conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)

llm = ChatOllama(model="qwen2.5:3b")

system_prompt="""
You are a helpful assistant.
Use get_date tool when the user is asking about today's date.
Use web search for up-to-date answers.
"""

search_tool = TavilySearch(max_results=2, topic="general")

agent = create_agent(
    model=llm,
#    tools=[TavilySearch(max_results=5, topic="general"), get_date],
    tools=[get_date, search_tool],
    system_prompt=system_prompt,
)

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()) )
    gr.ChatInterface(fn=partial(chat, agent=agent), additional_inputs=[thread_id])

demo.launch()
