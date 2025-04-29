def make_response_prefix(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type, new_subq=None, new_suba=None, new_ost_step=None
) -> str:
    if node_type in [Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER]:
        response_prefix = ""
        answer_marker = "The answer is"  # todo: hard code "The answer is"
        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            response_prefix += solution_step["subanswer"]["text"].split(answer_marker)[0]
            response_prefix += " "

        if new_subq is not None and new_suba is not None:
            response_prefix += new_suba.split(answer_marker)[0]

        response_prefix = response_prefix.strip(" ")
    elif node_type is Node_Type.OST_STEP:
        response_prefix = ""

        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        if "ost_step" in last_tuple_recording.keys():
            for step_id, step_text in last_tuple_recording["ost_step"].items():
                response_prefix += step_text + " "

        if new_ost_step is not None:
            response_prefix += new_ost_step

        response_prefix = response_prefix.strip(" ")
    elif node_type is None and solution_trace is None:
        response_prefix = ""
    else:
        raise ValueError(f"Invalid node type: {node_type}.")
    think = "Let's think step by step. "
    return think + response_prefix if think not in response_prefix else response_prefix

def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None
    if "Now we can answer" in subquestion:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True
    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower():
        return True
    return False
def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem

def concat_subqs_and_subas(solution_trace: Dict[int, Dict[str, str]], question_index: int) -> Tuple[str, int]:
    """Return: concatenated subqs and suba, next subquestion id"""
    solution_trace_str = ""

    for subquestion_id, solution_step in solution_trace.items():
        if subquestion_id == 0:
            continue

        assert subquestion_id > 0
        assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

        solution_trace_str += f"Question {question_index}." + str(subquestion_id) + ": " + solution_step["subquestion"]
        solution_trace_str += "\n"
        solution_trace_str += (
            f"Answer {question_index}." + str(subquestion_id) + ": " + solution_step["subanswer"]["text"]
        )
        solution_trace_str += "\n"

    next_subquestion_id = int(sorted(solution_trace.keys())[-1]) + 1
    return solution_trace_str, next_subquestion_id


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # no one-step thought step yet
        return "", 1


def concat_subqs_subas_as_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated subqs and subas as one-step thought steps, next one-step thought step id"""
    """Example solution trace (subq suba):
    {
        "0": {
            "user_question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "ost_step": {}
        },
        "1": {
            "subquestion": " How many eggs do the ducks lay each day?",
            "subanswer": {
                "text": "The ducks lay 16 eggs per day. The answer is 16.",
                "value": 1.0
            },
            "ost_step": {}
        },
        "2": {
            "subquestion": " How many eggs does Janet eat or use for baking muffins?",
            "subanswer": {
                "text": "Janet eats 3 eggs for breakfast and uses 4 eggs for baking muffins. That's a total of 3 + 4 = 7 eggs. The answer is 7.",
                "value": 1.0
            },
            "ost_step": {}
        },
        "3": {
            "subquestion": " Now we can answer the question: How much in dollars does she make every day at the farmers' market?",
            "subanswer": {
                "text": "Since the ducks lay 16 eggs per day and Janet eats/use 7 eggs, she has 16 - 7 = 9 eggs left to sell at the market. Each egg is sold for $2, so she makes 9 * 2 = 18 dollars. The answer is 18.",
                "value": 1.0
            },
            "ost_step": {}
        }
    },

    Expected output:
        subqs_subas_as_ost_steps_str:

            Step 1: The ducks lay 16 eggs per day.
            Step 2: Janet eats 3 eggs for breakfast and uses 4 eggs for baking muffins. That's a total of 3 + 4 = 7 eggs.
            Step 3: Since the ducks lay 16 eggs per day and Janet eats/use 7 eggs, she has 16 - 7 = 9 eggs left to sell at the market. Each egg is sold for $2, so she makes 9 * 2 = 18 dollars.

        next_ost_step_id: 4
    """
    subqs_subas_as_ost_steps_str = ""
    step_id = 1
    while step_id in solution_trace:
        if "subanswer" in solution_trace[step_id]:
            print(solution_trace[step_id])
            print(solution_trace[step_id]["subanswer"]["text"])
            match = re.search(r"(.+?\.) The answer is", solution_trace[step_id]["subanswer"]["text"])
            if match:
                step_text = match.group(1).strip()
            else:
                step_text = solution_trace[step_id]["subanswer"]["text"].strip()
            subqs_subas_as_ost_steps_str += f"Step {step_id}: " + step_text + "\n"
            step_id += 1
        else:
            # not subquestions yet
            return "", 1
    return subqs_subas_as_ost_steps_str, 1