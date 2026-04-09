from datetime import datetime

from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tools import parse_langchain_agent_response

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_date():
    """
    returns the current date and time
    """
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    return now

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

system_prompt="""
You are a helpful assistant.
Use get_date tool if user is asking about today's date
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)

user_query = input("Enter a query: ")
user_query = user_query.lower()

agent_input = {"messages": [{"role":"user", "content": user_query}]}

result = agent.invoke(agent_input)
parsed = parse_langchain_agent_response(result)

print(parsed["final_answer"])
print(parsed["tool_calls"])
print(parsed["tool_outputs"])
