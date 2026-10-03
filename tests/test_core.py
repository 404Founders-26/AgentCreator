"""Fast tests (no GPU, no internet): verifiers, presets, and generate -> filter against a fake teacher.

Run:  python -m pytest -q tests
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from distiller import presets
from distiller.config import load_config
from distiller.filter import run_filter
from distiller.generate import run_generate
from distiller.runlog import Run, read_json, read_jsonl
from distiller.verifiers import extract_code, extract_final_answer, run_python_tests, split_think, verify

# ---------------------------------------------------------------- verifiers

def test_split_think():
    assert split_think("<think>hmm</think>\nAnswer: 4") == ("hmm", "Answer: 4")
    assert split_think("hmm</think>done") == ("hmm", "done")
    assert split_think("<think>never closed") == ("never closed", "")
    assert split_think("plain") == ("", "plain")


def test_extract_code_prefers_python_block():
    text = "intro\n```bash\npip install x\n```\n```python\ndef f():\n    return 1\n```"
    assert extract_code(text).startswith("def f")


def test_python_tests_pass_fail_timeout():
    assert run_python_tests("def add(a,b): return a+b", ["assert add(1,2)==3"])[0]
    assert not run_python_tests("def add(a,b): return a-b", ["assert add(1,2)==3"])[0]
    ok, why = run_python_tests("while True: pass", ["assert True"], timeout=2)
    assert not ok and why == "timeout"


@pytest.mark.parametrize("text,expected", [
    ("so... Answer: 1,250", "1250"), ("Answer: $18.00", "18"), ("the result is \\boxed{42}", "42"),
    ("we get 3 then 7", "7"), ("Answer: -2.5", "-2.5"),
])
def test_final_answer(text, expected):
    assert extract_final_answer(text) == expected


def test_verify_numeric_and_code():
    rec = {"verifier": "numeric_answer", "answer": "72"}
    assert verify(rec, "work...\nAnswer: 72")[0]
    assert not verify(rec, "Answer: 70")[0]
    code_rec = {"verifier": "python_tests", "tests": ["assert sq(3)==9"], "setup": ""}
    assert verify(code_rec, "```python\ndef sq(x):\n    return x*x\n```")[0]
    assert verify(code_rec, "no code")[1] == "no_code_block"


# ---------------------------------------------------------------- fake teacher + pipeline

ANSWERS = {
    "square": "Use multiplication.\n```python\ndef sq(x):\n    return x * x\n```",
    "broken": "```python\ndef bad(x):\n    return x + 1\n```",          # fails its test
    "math": "Answer: 12",
}


class FakeTeacher(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"data": [{"id": "fake-9b"}]})

    counter = [0]

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        if "sub-topics" in user:
            content = json.dumps(["joins", "indexes", "window functions", "schema design", "query plans"])
        elif "Return ONLY a JSON array" in user:  # prompt writing for custom specialties
            import re as _re
            n = int(_re.search(r"Write (\d+)", user).group(1))
            FakeTeacher.counter[0] += 1
            c = FakeTeacher.counter[0]
            content = "```json\n" + json.dumps([f"SQL task number {c}-{i}: explain something useful" for i in range(n)]) + "\n```"
        elif "<candidate_answer>" in user:
            content = "The candidate is fine.\nScore: 8"
        else:
            key = next((k for k in ANSWERS if k in user), "math")
            content = ANSWERS[key]
            if key == "math":
                content = "<think>3 * 4 = 12</think>\n" + content
        self._send({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 20}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def teacher_url():
    srv = HTTPServer(("127.0.0.1", 0), FakeTeacher)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture()
def tiny_preset(tmp_path, monkeypatch):
    seeds = tmp_path / "seeds.jsonl"
    rows = [
        {"id": "a", "prompt": "Write square function sq", "tests": ["assert sq(4)==16"]},
        {"id": "b", "prompt": "Write broken function bad", "tests": ["assert bad(1)==5"]},
        {"id": "c", "prompt": "What is 3*4? (math)", "answer": "12"},
    ]
    seeds.write_text("\n".join(json.dumps(r) for r in rows))
    pdir = tmp_path / "presets"
    pdir.mkdir()
    (pdir / "tiny.yaml").write_text(f"""
name: tiny
keep_thinking: true
system_prompt: be good
sources:
  - name: code
    type: jsonl
    path: {seeds}
    verifier: python_tests
    limit: 2
  - name: missing
    type: hf
    path: this/does-not-exist-xyz
  - name: math
    type: jsonl
    path: {seeds}
    verifier: numeric_answer
""")
    monkeypatch.setattr(presets, "PRESET_DIR", pdir)
    return "tiny"


def test_generate_and_filter(tmp_path, teacher_url, tiny_preset, monkeypatch):
    monkeypatch.setattr("distiller.presets._iter_hf", lambda spec: (_ for _ in ()).throw(RuntimeError("offline")))
    cfg = load_config(overrides={"specialty": tiny_preset, "teacher": {"base_url": teacher_url, "model": "fake-9b", "retries": 0},
                                 "data": {"samples_per_prompt": 2, "keep_per_prompt": 1, "min_chars": 5}})
    run = Run(tmp_path / "run", echo=False)
    res = run_generate(cfg, run)
    assert res["new"] == 2 * 5  # 2 code prompts + 3 math prompts, 2 samples each
    # second call resumes and does nothing
    assert run_generate(cfg, run)["new"] == 0

    stats = run_filter(cfg, run)
    train = list(read_jsonl(run.dir / "data" / "train.jsonl"))
    ids = {r["id"] for r in train}
    assert "code-a" in ids and "code-b" not in ids          # broken code rejected by its test
    assert stats["reasons"]["tests_failed"] == 2
    math_rows = [r for r in train if r["source"] == "math"]
    # the math answer only matches row c's ground truth (12); rows a/b have no answer -> rejected
    assert {r["id"] for r in math_rows} == {"math-c"}
    assert math_rows[0]["assistant"].startswith("<think>\n3 * 4 = 12\n</think>")
    assert read_json(run.dir / "data" / "stats.json")["train_examples"] == len(train)


def test_check_teacher(teacher_url):
    from distiller.teacher import check_teacher

    cfg = load_config(overrides={"teacher": {"base_url": teacher_url, "model": "fake-9b"}})
    res = check_teacher(cfg)
    assert res["ok"] and res["models"] == ["fake-9b"] and res["has_reasoning"] is True  # fake answers with a <think> block


def test_real_presets_parse():
    names = {p["name"] for p in presets.list_presets()}
    assert {"coding", "reasoning"} <= names
    for n in names:
        p = presets.load_preset(n)
        assert p["sources"] and p["eval"]["verifier"]


# ---------------------------------------------------------------- custom specialties

def test_custom_specialty_generate_filter(tmp_path, teacher_url, monkeypatch):
    pdir = tmp_path / "presets"
    pdir.mkdir()
    monkeypatch.setattr(presets, "PRESET_DIR", pdir)
    preset = presets.save_custom_preset({
        "title": "SQL Expert", "description": "Writing and explaining SQL queries for PostgreSQL",
        "examples": ["top customers per month"], "answer_check": "none", "num_prompts": 40, "eval_holdout": 6,
    })
    assert preset["name"] == "sql-expert" and (pdir / "custom" / "sql-expert.yaml").exists()
    assert any(p["name"] == "sql-expert" and p["custom"] for p in presets.list_presets())

    cfg = load_config(overrides={"specialty": "sql-expert", "teacher": {"base_url": teacher_url, "model": "fake-9b", "retries": 0},
                                 "data": {"min_chars": 5}})
    run = Run(tmp_path / "run", echo=False)
    res = run_generate(cfg, run)
    prompts = list(read_jsonl(run.dir / "data" / "synthetic_prompts.jsonl"))
    assert len(prompts) == 40 and len({p["prompt"] for p in prompts}) == 40
    assert res["requests"] == 40  # 35 train + 5 held out (holdout is capped at 10%, min 5)
    stats = run_filter(cfg, run)
    refs = list(read_jsonl(run.dir / "data" / "eval_refs.jsonl"))
    train_ids = {r["id"] for r in read_jsonl(run.dir / "data" / "train.jsonl")}
    assert len(refs) == 5 and not (train_ids & {r["id"] for r in refs})   # no leakage
    assert stats["eval_references"] == 5
    # resuming writes no new prompts
    assert run_generate(cfg, run)["new"] == 0


def test_judge_scores(teacher_url):
    from distiller.synth import judge_outputs, parse_json_list, parse_score

    assert parse_score("blah\nScore: 7") == 7 and parse_score("<think>Score: 2</think>I give it 9/10") == 9
    assert parse_json_list('<think>x</think>```json\n["a", "b"]\n```') == ["a", "b"]
    assert parse_json_list("1. first\n2. second") == ["first", "second"]
    cfg = load_config(overrides={"teacher": {"base_url": teacher_url, "model": "fake-9b"}})
    out = judge_outputs(cfg, [{"user": "q", "reference": "r", "candidate": "c"}] * 3, print)
    assert [o["judge_score"] for o in out] == [8, 8, 8]


def test_catalog_estimates():
    from distiller.catalog import guess_params, student_advice

    assert guess_params("Qwen/Qwen3.5-2B") == 2.0
    assert guess_params("someone/MyModel-1.5B-Instruct") == 1.5
    small = student_advice("Qwen/Qwen3.5-0.8B", False, 2048, 2, 8)
    big = student_advice("Qwen/Qwen3.5-4B", False, 2048, 2, 8)
    assert small["estimated_vram_gb"] < 8 and not small["warnings"]
    assert big["warnings"]


# ---------------------------------------------------------------- Windows-specific behaviour (simulated)

def test_write_json_retries_when_windows_locks_file(tmp_path, monkeypatch):
    import os as _os
    from distiller import runlog

    real = _os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("[WinError 5] Access is denied")
        return real(src, dst)

    monkeypatch.setattr(runlog.os, "replace", flaky)
    runlog.write_json(tmp_path / "state.json", {"a": 1})
    assert runlog.read_json(tmp_path / "state.json") == {"a": 1} and calls["n"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_name_avoids_windows_reserved_names():
    from distiller.config import safe_name

    assert safe_name("CON") == "CON-run" and safe_name("nul.txt") == "nul.txt-run"
    assert safe_name("my run: v2/<test>") == "my-run--v2--test"


def test_find_llama_binary_in_unzipped_release(tmp_path):
    from distiller.export import find_llama_binary

    nested = tmp_path / "llama-b9999-bin-win-cuda" / "bin"
    nested.mkdir(parents=True)
    (nested / "llama-quantize").write_text("")
    assert find_llama_binary(tmp_path, "llama-quantize") == str(nested / "llama-quantize")
    assert find_llama_binary(tmp_path, "llama-bench") is None or "llama-bench" in find_llama_binary(tmp_path, "llama-bench")


def test_code_runner_handles_unicode_output():
    ok, _ = run_python_tests("def f():\n    print('héllo → 世界 ✓')\n    return 1", ["assert f() == 1"])
    assert ok


def test_log_stream_handles_crlf_and_partial_lines(tmp_path, monkeypatch):
    """The SSE log tail must cope with Windows \\r\\n line endings and lines written in pieces."""
    import asyncio

    import server.app as srv

    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "run.log").write_bytes(b"first line\r\nsecond \xe2\x86\x92 line\r\n")
    monkeypatch.setattr(srv, "run_path", lambda name: run_dir)

    class Req:
        n = 0

        async def is_disconnected(self):
            Req.n += 1
            if Req.n == 1:
                with open(run_dir / "run.log", "ab") as f:
                    f.write(b"third li")
            if Req.n == 2:
                with open(run_dir / "run.log", "ab") as f:
                    f.write(b"ne\r\n")
            return Req.n > 3

    async def collect():
        resp = await srv.logs("r1", Req())
        return [c async for c in resp.body_iterator]

    real_sleep = asyncio.sleep
    monkeypatch.setattr(srv.asyncio, "sleep", lambda s: real_sleep(0))
    chunks = [c if isinstance(c, str) else c.decode() for c in asyncio.run(collect())]
    lines = [c[len("data: "):].strip("\n") for c in chunks if c.startswith("data: ")]
    assert lines == ["first line", "second → line", "third line"]


def test_ollama_native_api_think_switch_and_vram():
    """Against a fake Ollama: native /api/chat is used, thinking can be switched off, VRAM placement is read."""
    from distiller.teacher import Teacher

    seen = []

    class FakeOllama(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/version":
                return self._send({"version": "0.12.0"})
            if self.path == "/api/ps":
                return self._send({"models": [{"name": "qwen3.5:9b", "size": 8_000_000_000, "size_vram": 6_000_000_000}]})
            self._send({"data": [{"id": "qwen3.5:9b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            think = body.get("think")
            self._send({"message": {"content": "def f(): pass", "thinking": "hmm" if think else ""},
                        "done_reason": "stop", "eval_count": 100, "eval_duration": 2_000_000_000, "prompt_eval_count": 20})

    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = load_config(overrides={"teacher": {"base_url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "qwen3.5:9b"}})
        t = Teacher(cfg)
        res = t.chat([{"role": "user", "content": "x"}], think=False)
        assert t.is_ollama() and seen[-1]["think"] is False and seen[-1]["options"]["num_ctx"] == 4096
        assert res["gen_tokens_per_second"] == 50.0 and res["reasoning"] == ""
        assert t.chat([{"role": "user", "content": "x"}], think=True)["reasoning"] == "hmm"
        place = t.gpu_placement()
        assert place["gpu_fraction"] == 0.75
        t.close()
    finally:
        srv.shutdown()
