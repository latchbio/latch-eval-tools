from latch_eval_tools.answer_extraction import extract_answer_from_pi_trajectory


def test_extract_answer_from_pi_message_end_text_block():
    trajectory = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": '<EVAL_ANSWER>{"count": 7}</EVAL_ANSWER>',
                    }
                ],
            },
        }
    ]

    assert extract_answer_from_pi_trajectory(trajectory) == {"count": 7}


def test_extract_answer_from_pi_trajectory_ignores_user_messages():
    trajectory = [
        {
            "type": "message_end",
            "message": {
                "role": "user",
                "content": '<EVAL_ANSWER>{"count": 7}</EVAL_ANSWER>',
            },
        }
    ]

    assert extract_answer_from_pi_trajectory(trajectory) is None
