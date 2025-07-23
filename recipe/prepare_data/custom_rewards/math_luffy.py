from math_verify import parse, verify

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def rfn(solution_str: str, ground_truth: str, **kwargs):
    labels = labeling_responses([solution_str,], ground_truth)
    if labels[0] == True:
        return {"score": 1.0}
    else:
        return {"score": 0.0}