import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile, TemporaryDirectory

from verl.utils.reward_score import _default_compute_score

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30
_FORMAT_REWARD = 0.1
_ANSWER_REWARD = 1.0
_CLI_ARG_SIZE_LIMIT = 1024 * 3
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CODE_PATTERN = re.compile(r"```(?:[\w.+-]+)?\s*\n(.*?)```", re.DOTALL)
_CODE_DATA_SOURCES = {
    "code",
    "apps",
    "codecontests",
    "taco",
    "mbpp",
    "humaneval",
    "livecodebench",
    "lcb",
}
_FIREJAIL_ALLOWED_ENV_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
)


def _normalize_extra_info(extra_info):
    if hasattr(extra_info, "item"):
        try:
            extra_info = extra_info.item()
        except Exception:
            pass
    return extra_info if isinstance(extra_info, dict) else {}


def _normalize_data_source(data_source):
    if hasattr(data_source, "item"):
        try:
            data_source = data_source.item()
        except Exception:
            pass
    return data_source


def _is_code_source(data_source):
    data_source = _normalize_data_source(data_source)
    return isinstance(data_source, str) and data_source.lower() in _CODE_DATA_SOURCES


def _validate_response_structure(response_text):
    if response_text is None:
        return False
    return bool(_ANSWER_PATTERN.search(response_text))


def _extract_answer_block(response_text):
    matches = list(_ANSWER_PATTERN.finditer(response_text))
    if matches:
        return matches[-1].group(1).strip()
    return response_text


def _extract_code(response_text):
    answer_text = _extract_answer_block(response_text)
    code_blocks = _CODE_PATTERN.findall(answer_text)
    if not code_blocks:
        return ""
    return "\n\n".join(block.strip() for block in code_blocks if block.strip()).strip()


def _load_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        return ground_truth
    if hasattr(ground_truth, "item"):
        try:
            ground_truth = ground_truth.item()
        except Exception:
            pass
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    raise TypeError(f"Unsupported ground truth type: {type(ground_truth)!r}")


def _build_firejail_env():
    env = {}
    for key in _FIREJAIL_ALLOWED_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.setdefault("HOME", os.path.expanduser("~"))
    env.setdefault("PATH", "/usr/bin:/bin")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def _firejail_timeout_arg(timeout_seconds):
    timeout_seconds = max(1, int(timeout_seconds))
    hours, remainder = divmod(timeout_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"--timeout={hours:02d}:{minutes:02d}:{seconds:02d}"


def _run_subprocess(command, stdin=None, cwd=None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    try:
        result = subprocess.run(
            command,
            input=stdin.encode() if stdin is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_build_firejail_env(),
            cwd=cwd,
            check=False,
            timeout=timeout + 5,
        )
    except subprocess.TimeoutExpired:
        return False, _ERROR_MSG_PREFIX + f"Timed out after {timeout} seconds"
    except Exception as exc:
        return False, _ERROR_MSG_PREFIX + str(exc)

    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace").strip()
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def _code_exec_firejail(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",
        "--rlimit-as=4096m",
        _firejail_timeout_arg(timeout),
    ]

    if pytest is not None:
        with TemporaryDirectory() as tmpdir:
            if stdin is not None:
                raise ValueError("STDIN is not supported with pytest execution")
            with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as handle:
                handle.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as handle:
                handle.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
            return _run_subprocess(command, cwd=tmpdir, timeout=timeout)

    if len(code) < _CLI_ARG_SIZE_LIMIT:
        command.extend(["python3", "-c", code])
        return _run_subprocess(command, stdin=stdin, timeout=timeout)

    with NamedTemporaryFile(mode="w", suffix=".py") as handle:
        handle.write(code)
        handle.flush()
        command.insert(4, f"--whitelist={handle.name}")
        command.extend(["python3", handle.name])
        return _run_subprocess(command, stdin=stdin, timeout=timeout)


def _select_executor():
    executor_name = os.environ.get("CODER1_EXEC", "firejail").strip().lower()
    if executor_name != "firejail":
        print(f"Unsupported CODER1_EXEC={executor_name}; falling back to firejail")
    return _code_exec_firejail


_CODE_EXEC = _select_executor()


def _remote_check_stdio(code, stdin, stdout, timeout):
    success, output = _CODE_EXEC(code=code, stdin=stdin, timeout=timeout)
    return success, output, stdin, stdout


def _exec_functional(solution_code, functional_tests, timeout):
    success, output = _CODE_EXEC(solution_code + "\n" + functional_tests, timeout=timeout)
    return success, output


def _exec_pytest(solution_code, pytest_code, timeout):
    success, output = _CODE_EXEC(solution_code, pytest=pytest_code, timeout=timeout)
    return success, output


def _exec_stdio(solution_code, inputs, outputs, timeout):
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(inputs)))) as executor:
        futures = [
            executor.submit(_remote_check_stdio, solution_code, stdin, stdout, timeout)
            for stdin, stdout in zip(inputs, outputs)
        ]
        for future in as_completed(futures):
            success, output, stdin, stdout = future.result()
            if not success or output.strip() != stdout.strip():
                return False, {"input": stdin, "expected": stdout, "actual": output}
    return True, {}


def _score_code(solution_text, ground_truth, extra_info, timeout, format_reward, answer_reward):
    response_text = solution_text or ""
    if not _validate_response_structure(response_text):
        return {
            "score": -(answer_reward + format_reward),
            "acc": 0.0,
            "formatted": 0.0,
            "passed": 0.0,
        }

    solution_code = _extract_code(response_text)
    if not solution_code:
        return {
            "score": -(answer_reward + format_reward),
            "acc": 0.0,
            "formatted": 0.0,
            "passed": 0.0,
        }

    try:
        ground_truth_obj = _load_ground_truth(ground_truth)
    except Exception:
        return {
            "score": 0.0,
            "acc": 0.0,
            "formatted": 1.0,
            "passed": 0.0,
            "valid_ground_truth": 0.0,
        }

    try:
        if "functional" in ground_truth_obj:
            passed, _ = _exec_functional(solution_code, ground_truth_obj["functional"], timeout=timeout)
        elif "pytest" in ground_truth_obj:
            passed, _ = _exec_pytest(solution_code, ground_truth_obj["pytest"], timeout=timeout)
        elif "inputs" in ground_truth_obj and "outputs" in ground_truth_obj:
            passed, _ = _exec_stdio(solution_code, ground_truth_obj["inputs"], ground_truth_obj["outputs"], timeout=timeout)
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "formatted": 1.0,
                "passed": 0.0,
                "valid_ground_truth": 0.0,
            }
    except Exception as exc:
        prompt_text = extra_info.get("prompt") or extra_info.get("question") or ""
        print(f"code reward execution failed: {exc}")
        if prompt_text:
            print(prompt_text)
        passed = False

    reward = format_reward + answer_reward if passed else format_reward
    return {
        "score": reward,
        "acc": float(passed),
        "formatted": 1.0,
        "passed": float(passed),
        "valid_ground_truth": 1.0,
    }


def rfn(
    solution_str: str,
    ground_truth,
    data_source=None,
    response_str=None,
    extra_info=None,
    timeout=_DEFAULT_TIMEOUT_SECONDS,
    format_reward=_FORMAT_REWARD,
    answer_reward=_ANSWER_REWARD,
    **kwargs,
):
    if not _is_code_source(data_source):
        score = _default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return {"score": float(score), "acc": float(score)}

    return _score_code(
        solution_text=response_str or solution_str,
        ground_truth=ground_truth,
        extra_info=_normalize_extra_info(extra_info),
        timeout=timeout,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )


def rfn_val(*args, **kwargs):
    return rfn(*args, **kwargs)


def compute_score(*args, **kwargs):
    return rfn(*args, **kwargs)


def compute_score_val(*args, **kwargs):
    return rfn_val(*args, **kwargs)
