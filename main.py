import uuid
from functools import partial

from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from tools import chat, get_date

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)
llm = ChatOllama(model="qwen2.5:3b")

system_prompt="""
You are a helpful assistant. Answer all user queries.
Use get_date tool if user is asking about today's date
"""

agent = create_agent(model=llm,
                     tools=[get_date],
                     system_prompt=system_prompt,
                     checkpointer=checkpointer)

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()) )
    gr.ChatInterface(fn=partial(chat, agent=agent), additional_inputs=[thread_id])

demo.launch()
