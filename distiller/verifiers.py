"""Checks that decide whether a teacher (or student) answer is good enough to keep.

python_tests   - extract the code block and run the record's unit tests against it
numeric_answer - compare the final "Answer: x" with the ground-truth number
has_code       - answer must contain a fenced code block
none           - only length / format checks
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def split_think(text: str) -> tuple[str, str]:
    """Return (reasoning, answer) from text that may contain <think>...</think>."""
    text = text or ""
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), (text[: m.start()] + text[m.end():]).strip()
    if "</think>" in text:  # opening tag was in the prompt
        head, _, tail = text.partition("</think>")
        return head.replace("<think>", "").strip(), tail.strip()
    if "<think>" in text:  # never closed: model ran out of tokens while thinking
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def extract_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text or "")
    if not blocks:
        return None
    py = [code for lang, code in blocks if lang.lower() in ("python", "py", "python3", "")]
    pool = py or [code for _, code in blocks]
    return max(pool, key=len).strip()


def _limit_resources():  # POSIX only: cap memory and CPU of the test process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    except Exception:
        pass


def run_python_tests(code: str, tests: list[str], setup: str = "", timeout: float = 10) -> tuple[bool, str]:
    """Run code + asserts in a separate, isolated Python process."""
    program = "\n\n".join([setup or "", code, "\n".join(tests), "print('__ALL_TESTS_PASSED__')"])
    # ignore_cleanup_errors: on Windows a just-killed process can still hold the file for a moment
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        if os.name == "nt":  # Windows needs these for Python (and sockets/random) to start at all
            for k in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
            env["TEMP"] = env["TMP"] = tmp
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as e:  # e.g. antivirus blocking the temp file on Windows
            return False, f"could not run: {e}"[:200]
        if proc.returncode == 0 and "__ALL_TESTS_PASSED__" in proc.stdout:
            return True, "passed"
        err = (proc.stderr or "").strip().splitlines()
        return False, (err[-1] if err else f"exit {proc.returncode}")[:200]


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace("%", "").replace("\\!", "")
    m = _NUM_RE.findall(s)
    if not m:
        return None
    n = m[0].replace(",", "")
    try:
        val = float(n)
    except ValueError:
        return None
    if val == int(val):
        return str(int(val))
    return f"{val:.6g}"


def extract_final_answer(text: str) -> str | None:
    text = text or ""
    hits = re.findall(r"Answer\s*[:：]\s*\**\s*([^\n]+)", text, flags=re.IGNORECASE)
    if hits:
        return normalize_number(hits[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def verify(record: dict, answer_text: str, *, execute_code: bool = True, timeout: float = 10) -> tuple[bool, str]:
    """answer_text should already have <think> removed."""
    kind = record.get("verifier", "none")
    if kind == "python_tests":
        code = extract_code(answer_text)
        if not code:
            return False, "no_code_block"
        if not record.get("tests"):
            return True, "no_tests"
        if not execute_code:
            try:
                compile(code, "<candidate>", "exec")
                return True, "compiled"
            except SyntaxError:
                return False, "syntax_error"
        ok, why = run_python_tests(code, record["tests"], record.get("setup", ""), timeout)
        return ok, ("passed" if ok else "tests_failed")
    if kind == "numeric_answer":
        gold = normalize_number(record.get("answer"))
        got = extract_final_answer(answer_text)
        if got is None:
            return False, "no_answer"
        return (got == gold), ("correct" if got == gold else "wrong_answer")
    if kind == "has_code":
        return (extract_code(answer_text) is not None), ("has_code" if extract_code(answer_text) else "no_code_block")
    return True, "unchecked"
