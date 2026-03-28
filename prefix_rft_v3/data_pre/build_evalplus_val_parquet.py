import argparse
import inspect
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are a helpful programming assistant. The user will ask you a question and you as the assistant solve it. "
    "The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. "
    "The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."
)

HUMANEVAL_TEST_TEMPLATE = """\
{imports}

{aux_fn}

def check(candidate):
    inputs = {inputs}
    results = {results}
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        {assertion}
"""

HUMANEVAL_CROSSCHECK_TEMPLATE = """\
{aux_fn}

{ref_func}

def check(candidate):
    inputs = {inputs}
    for i, inp in enumerate(inputs):
        assertion(candidate(*inp), ref_func(*inp), {atol})
"""

MBPP_TEST_TEMPLATE = """\
{imports}

{aux_fn}

inputs = {inputs}
results = {results}
for i, (inp, exp) in enumerate(zip(inputs, results)):
    {assertion}
"""

MBPP_CROSSCHECK_TEMPLATE = """\
import numpy as np
from math import inf

{aux_fn}

{ref_func}

inputs = {inputs}
for i, inp in enumerate(inputs):
    assertion({entry_point}(*inp), ref_func(*inp), {atol})
"""


def _add_evalplus_to_path(evalplus_root: Path) -> None:
    root_str = str(evalplus_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _deduplicate(inputs, results):
    assert len(inputs) == len(results)
    unique_input_strs = {f"{item}" for item in inputs}

    new_inputs = []
    new_results = []
    for inp, res in zip(inputs, results):
        inp_str = f"{inp}"
        if inp_str in unique_input_strs:
            new_inputs.append(inp)
            new_results.append(res)
            unique_input_strs.remove(inp_str)
    return new_inputs, new_results


def _normalize_evalplus_inputs(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return []
    if value is None:
        return []
    raise TypeError(f"Unsupported EvalPlus input container: {type(value)}")


def _get_groundtruth(problems, hashcode, tasks_only_output_not_none, trusted_exec):
    cache_root = Path.home() / ".cache" / "evalplus"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{hashcode}.pkl"
    if cache_file.exists():
        import pickle

        with cache_file.open("rb") as handle:
            return pickle.load(handle)

    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle

    import pickle

    with cache_file.open("wb") as handle:
        pickle.dump(expected_output, handle)
    return expected_output


def _build_humaneval_assertion_fn(is_floats_src: str) -> str:
    return f"""\
import numpy as np

{is_floats_src}

def assertion(out, exp, atol):
    exact_match = out == exp

    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact_match and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert exact_match
"""


def _build_mbpp_assertion_fn(is_floats_src: str) -> str:
    return f"""\
import numpy as np

{is_floats_src}

def assertion(out, exp, atol):
    exact_match = out == exp

    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact_match and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert exact_match, f\"out: {{out}}, exp: {{exp}}\"
"""


def _synthesize_humaneval_test(task_id, entry_point, inputs, results, ref_func, atol, is_floats_src, poly_src):
    assertion_fn = _build_humaneval_assertion_fn(is_floats_src)

    if entry_point in (
        "tri",
        "string_sequence",
        "starts_one_ends",
        "make_a_pile",
        "special_factorial",
        "all_prefixes",
    ):
        return HUMANEVAL_CROSSCHECK_TEMPLATE.format(
            aux_fn=assertion_fn,
            inputs=inputs,
            ref_func=ref_func.replace(f" {entry_point}(", " ref_func("),
            atol=atol,
        )

    imports = set()
    aux_fn = assertion_fn
    assertion = f"assertion(candidate(*inp), exp, {atol})"

    if entry_point == "find_zero":
        imports.add("import math")
        aux_fn = poly_src + "\n"
        assertion = f"assert _poly(*candidate(*inp), inp) <= {atol}"

    return HUMANEVAL_TEST_TEMPLATE.format(
        imports="\n".join(imports),
        aux_fn=aux_fn,
        inputs=inputs,
        results=results,
        assertion=assertion,
    )


def _synthesize_mbpp_test(
    task_id,
    entry_point,
    inputs,
    results,
    ref_func,
    atol,
    is_floats_src,
    output_not_none_tasks,
    output_set_eq_tasks,
    digit_distance_src,
    surface_area_src,
):
    assertion_fn = _build_mbpp_assertion_fn(is_floats_src)

    if entry_point in ("combinations_colors", "freq_count", "get_coordinates"):
        return MBPP_CROSSCHECK_TEMPLATE.format(
            aux_fn=assertion_fn,
            inputs=inputs,
            ref_func=ref_func.replace(f" {entry_point}(", " ref_func("),
            entry_point=entry_point,
            atol=atol,
        )

    imports = {"import numpy as np", "from math import inf"}
    aux_fn = assertion_fn
    assertion = f"assertion({entry_point}(*inp), exp, {atol})"

    if entry_point in output_set_eq_tasks:
        aux_fn = f"""\
{is_floats_src}

def assertion(out, exp, atol):
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    out = set(out)
    exp = set(exp)
    if out != exp and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert out == exp, f\"out: {{out}}, exp: {{exp}}\"
"""
    elif entry_point in output_not_none_tasks:
        aux_fn = """\
def assertion(out, exp, atol):
    if isinstance(out, bool):
        exact_match = out == exp
    else:
        exact_match = exp == (out is not None)
    assert exact_match
"""
    elif entry_point == "surface_Area":
        imports.add("import math")
        aux_fn = surface_area_src + "\n" + """\
def assertion(out, exp_0, exp_1, atol):
    assert abs(out - exp_0) <= atol or abs(out - exp_1) <= atol
"""
        assertion = f"assertion(surface_Area(*inp), exp, _surface_Area(*inp), {atol})"
    elif entry_point == "digit_distance_nums":
        aux_fn = digit_distance_src + "\n" + """\
def assertion(out, exp_0, exp_1, atol):
    assert out == exp_0 or out == exp_1
"""
        assertion = f"assertion(digit_distance_nums(*inp), exp, _digit_distance_nums(*inp), {atol})"

    return MBPP_TEST_TEMPLATE.format(
        imports="\n".join(imports),
        aux_fn=aux_fn,
        inputs=inputs,
        results=results,
        assertion=assertion,
    )


def _build_chat_prompt(user_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_user_prompt(dataset_name: str, task: dict) -> str:
    task_prompt = task["prompt"].rstrip()
    contract = task.get("contract", "").strip()

    header = (
        "Please solve the programming task below using a self-contained Python code snippet in a markdown code block.\n"
        "Return the complete Python function including its signature."
    )
    body = [header, "", "Problem:", "```python", task_prompt, "```"]

    if contract:
        body.extend(["", "Input constraints:", "```python", contract, "```"])

    if dataset_name == "MBPPPlus":
        body.append("\nMake sure the returned function matches the examples in the prompt.")

    return "\n".join(body)


def _row_schema():
    prompt_struct = pa.struct([("content", pa.string()), ("role", pa.string())])
    reward_model_struct = pa.struct([("ground_truth", pa.string()), ("style", pa.string())])
    extra_info_struct = pa.struct(
        [
            ("dataset", pa.string()),
            ("index", pa.int64()),
            ("task_id", pa.string()),
            ("entry_point", pa.string()),
            ("prompt", pa.string()),
            ("reference", pa.string()),
            ("split", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("task_id", pa.string()),
            ("prompt", pa.list_(prompt_struct)),
            ("entry_point", pa.string()),
            ("test", pa.string()),
            ("data_source", pa.string()),
            ("ability", pa.string()),
            ("reward_model", reward_model_struct),
            ("extra_info", extra_info_struct),
            ("demos", pa.list_(pa.string())),
            ("demos_corr", pa.list_(pa.bool_())),
        ]
    )


def build_rows(evalplus_root: Path):
    _add_evalplus_to_path(evalplus_root)

    from evalplus.data import (
        get_human_eval_plus,
        get_human_eval_plus_hash,
        get_mbpp_plus,
        get_mbpp_plus_hash,
    )
    from evalplus.eval import is_floats
    from evalplus.eval._special_oracle import (
        MBPP_OUTPUT_NOT_NONE_TASKS,
        MBPP_OUTPUT_SET_EQ_TASKS,
        _digit_distance_nums,
        _poly,
        _surface_Area,
    )
    from evalplus.gen.util import trusted_exec

    is_floats_src = inspect.getsource(is_floats)
    poly_src = inspect.getsource(_poly)
    digit_distance_src = inspect.getsource(_digit_distance_nums)
    surface_area_src = inspect.getsource(_surface_Area)

    rows = []

    humaneval = get_human_eval_plus(mini=False)
    humaneval_outputs = _get_groundtruth(humaneval, get_human_eval_plus_hash(), [], trusted_exec)
    for index, (task_id, task) in enumerate(sorted(humaneval.items(), key=lambda item: int(item[0].split("/")[-1]))):
        inputs = _normalize_evalplus_inputs(task["base_input"]) + _normalize_evalplus_inputs(task["plus_input"])
        results = humaneval_outputs[task_id]["base"] + humaneval_outputs[task_id]["plus"]
        inputs, results = _deduplicate(inputs, results)
        full_reference = task["prompt"] + task["canonical_solution"]
        test_code = _synthesize_humaneval_test(
            task_id=task_id,
            entry_point=task["entry_point"],
            inputs=inputs,
            results=results,
            ref_func=full_reference,
            atol=task["atol"],
            is_floats_src=is_floats_src,
            poly_src=poly_src,
        )
        functional_test = test_code + f"\n\ncheck({task['entry_point']})"
        user_prompt = _build_user_prompt("HumanEvalPlus", task)
        rows.append(
            {
                "task_id": task_id,
                "prompt": _build_chat_prompt(user_prompt),
                "entry_point": task["entry_point"],
                "test": functional_test,
                "data_source": "humaneval",
                "ability": "code",
                "reward_model": {
                    "ground_truth": json.dumps({"functional": functional_test}),
                    "style": "rule",
                },
                "extra_info": {
                    "dataset": "HumanEvalPlus",
                    "index": index,
                    "task_id": task_id,
                    "entry_point": task["entry_point"],
                    "prompt": user_prompt,
                    "reference": full_reference,
                    "split": "test",
                },
                "demos": [],
                "demos_corr": [],
            }
        )

    mbpp = get_mbpp_plus(mini=False)
    mbpp_outputs = _get_groundtruth(mbpp, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS, trusted_exec)
    base_index = len(rows)
    for offset, (task_id, task) in enumerate(sorted(mbpp.items(), key=lambda item: int(item[0].split("/")[-1]))):
        inputs = _normalize_evalplus_inputs(task["base_input"]) + _normalize_evalplus_inputs(task["plus_input"])
        results = mbpp_outputs[task_id]["base"] + mbpp_outputs[task_id]["plus"]
        inputs, results = _deduplicate(inputs, results)
        full_reference = task["prompt"] + task["canonical_solution"]
        test_code = _synthesize_mbpp_test(
            task_id=task_id,
            entry_point=task["entry_point"],
            inputs=inputs,
            results=results,
            ref_func=task["canonical_solution"],
            atol=task["atol"],
            is_floats_src=is_floats_src,
            output_not_none_tasks=MBPP_OUTPUT_NOT_NONE_TASKS,
            output_set_eq_tasks=MBPP_OUTPUT_SET_EQ_TASKS,
            digit_distance_src=digit_distance_src,
            surface_area_src=surface_area_src,
        )
        user_prompt = _build_user_prompt("MBPPPlus", task)
        rows.append(
            {
                "task_id": task_id,
                "prompt": _build_chat_prompt(user_prompt),
                "entry_point": task["entry_point"],
                "test": test_code,
                "data_source": "mbpp",
                "ability": "code",
                "reward_model": {
                    "ground_truth": json.dumps({"functional": test_code}),
                    "style": "rule",
                },
                "extra_info": {
                    "dataset": "MBPPPlus",
                    "index": base_index + offset,
                    "task_id": task_id,
                    "entry_point": task["entry_point"],
                    "prompt": user_prompt,
                    "reference": full_reference,
                    "split": "test",
                },
                "demos": [],
                "demos_corr": [],
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evalplus-root",
        default="/mlx_devbox/users/zeyuhuang/playground/prefix-rft-rebuttal/evalplus",
        help="Path to the local evalplus repository.",
    )
    parser.add_argument(
        "--output-path",
        default="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code/val.parquet",
        help="Where to write the combined EvalPlus validation parquet.",
    )
    args = parser.parse_args()

    evalplus_root = Path(args.evalplus_root).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(evalplus_root)
    table = pa.Table.from_pylist(rows, schema=_row_schema())
    pq.write_table(table, output_path)
    print(f"Wrote {table.num_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
