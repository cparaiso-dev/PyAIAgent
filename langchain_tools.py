def parse_langchain_agent_response(result: dict) -> dict:
    messages = result.get("messages", [])

    final_answer = None
    tool_calls = []
    tool_outputs = []

    for msg in messages:
        # Collect tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_calls.append({
                    "tool_name": call.get("name"),
                    "tool_args": call.get("args"),
                    "tool_id": call.get("id"),
                })

        # Collect tool outputs
        if msg.__class__.__name__ == "ToolMessage":
            tool_outputs.append({
                "tool_name": getattr(msg, "name", None),
                "tool_output": msg.content,
                "tool_call_id": getattr(msg, "tool_call_id", None),
            })

    # Find final AI text answer
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            content = msg.content

            if isinstance(content, str) and content.strip():
                final_answer = content
                break

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    final_answer = " ".join(text_parts)
                    break

    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
    }