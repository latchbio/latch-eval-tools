import json
import re


def extract_answer_from_conversation(conversation: list[dict]) -> dict | None:
    """Extract the JSON answer from a conversation history.
    
    Looks for submit_response tool calls with EVAL_ANSWER tags in the summary.
    
    Args:
        conversation: List of message dicts from agent conversation
        
    Returns:
        Parsed JSON answer dict, or None if not found
    """
    for msg in reversed(conversation):
        if msg.get("type") != "anthropic_message" or msg.get("role") != "assistant":
            continue

        content = msg.get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") == "submit_response":
                    tool_input = block.get("input", {})
                    summary = tool_input.get("summary", "")

                    match = re.search(r'<EVAL_ANSWER>(.*?)</EVAL_ANSWER>', summary, re.DOTALL)
                    if match:
                        json_str = match.group(1).strip()
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            print(f"[grader] Failed to parse JSON from EVAL_ANSWER tags: {e}")
                            return None
    return None
