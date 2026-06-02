import json
import re


_EVAL_ANSWER_RE = re.compile(r"<EVAL_ANSWER>(.*?)</EVAL_ANSWER>", re.DOTALL)


def extract_answer_from_text(text: str) -> dict | None:
    match = _EVAL_ANSWER_RE.search(text)
    if match is None:
        return None

    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[grader] Failed to parse JSON from EVAL_ANSWER tags: {e}")
        return None


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            for key in ("text", "content", "summary"):
                value = block.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
    return "\n".join(parts)


def extract_answer_from_pi_trajectory(trajectory: list[dict]) -> dict | None:
    for event in reversed(trajectory):
        message = event.get("message")
        if isinstance(message, dict):
            if message.get("role") != "assistant":
                continue
            text = _content_to_text(message.get("content"))
        else:
            if event.get("role") != "assistant":
                continue
            text = _content_to_text(event.get("content"))

        answer = extract_answer_from_text(text)
        if answer is not None:
            return answer

    return None


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

                    answer = extract_answer_from_text(summary)
                    if answer is not None:
                        return answer
    return None
