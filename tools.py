from datetime import datetime

from langchain_tools import parse_langchain_agent_response


def get_date():
    """
    returns the current date and time
    """
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    return now


def chat(message, history, thread_id, agent):
    config = {'configurable': {"thread_id": thread_id}}
    agent_input = {"messages": [{"role": "user", "content": message}]}
    result = agent.invoke(agent_input, config)

    parsed = parse_langchain_agent_response(result)
    return parsed["final_answer"]
