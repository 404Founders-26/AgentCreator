// Read-only online demo of AgentCreator.
// The real app talks to a local Python server that drives the GPU; here every /api call is answered from
// demo-data.json - the results of a real run (qwen3.5:9b -> Qwen3.5-0.8B, coding, RTX 5050 Laptop 8 GB).
(function () {
  const READ_ONLY = "This is the read-only online demo - training needs your own NVIDIA GPU. " +
    "Get AgentCreator from GitHub (404Founders-26/AgentCreator) and double-click AgentCreator.bat to run it.";
  let DATA = null;
  const ready = fetch("demo-data.json").then((r) => r.json()).then((d) => (DATA = d));

  const json = (obj, status = 200) =>
    new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
  const denied = () => json({ detail: READ_ONLY }, 403);

  // same rough formula as distiller/catalog.py
  function advice(body) {
    const s = DATA.catalog.students.find((x) => x.id.toLowerCase() === String(body.model).toLowerCase());
    let p = s ? s.params : null;
    if (!p) {
      const m = String(body.model).toLowerCase().replace(/[-_]/g, " ").match(/(\d+(?:\.\d+)?)\s*([bm])\b/);
      if (m) p = m[2] === "m" ? parseFloat(m[1]) / 1000 : parseFloat(m[1]);
    }
    const fam = (String(body.model).toLowerCase().match(/qwen3\.5|qwen3|qwen2\.5|llama|gemma|mistral|phi|deepseek|smollm/) || ["other"])[0];
    const out = { params_b: p, estimated_vram_gb: null, family: fam, warnings: [] };
    if (p) {
      const need = p * (body.load_in_4bit ? 0.6 : 2.0) + 0.15 + 0.05 * p +
        0.35 * Math.sqrt(p) * ((body.max_seq_length || 2048) / 2048) * (body.batch_size || 2) + 0.8;
      out.estimated_vram_gb = Math.round(need * 10) / 10;
      if (need > 8 * 0.95) out.warnings.push(`~${out.estimated_vram_gb} GB needed but the GPU has 8 GB - ` +
        (body.load_in_4bit ? "lower max sequence length / batch size" : "turn on 4-bit or pick a smaller student"));
    }
    if (body.load_in_4bit && fam === "qwen3.5") out.warnings.push("Unsloth advises against 4-bit (QLoRA) training for Qwen3.5 models");
    return out;
  }

  const DOCTOR = [
    ["Python", "ok", "3.13 (Windows 11)"], ["Virtual environment", "ok", ".venv"],
    ["Core packages", "ok", "fastapi, uvicorn, httpx, pyyaml, datasets"],
    ["PyTorch + CUDA", "ok", "torch 2.11.0+cu128, NVIDIA GeForce RTX 5050 Laptop GPU (7.9 GB)"],
    ["bf16 support", "ok", "yes"], ["transformers", "ok", "5.5.0"], ["peft", "ok", "0.21.2"],
    ["Unsloth", "ok", "2026.9.14"], ["bitsandbytes", "ok", "0.50.2"],
    ["Teacher server", "ok", "http://localhost:11434/v1 - 6 model(s)"], ["Teacher model", "ok", "qwen3.5:9b installed"],
    ["GGUF export", "ok", "llama.cpp converter + llama-quantize, imported into Ollama"],
    ["Hugging Face", "ok", "reachable"], ["Code test runner", "ok", "can run generated code in a subprocess"],
  ].map(([name, status, detail]) => ({ name, status, detail, fix: "" }));

  const realFetch = window.fetch.bind(window);
  window.fetch = async function (input, init = {}) {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (!url.pathname.startsWith("/api/")) return realFetch(input, init);
    await ready;
    const path = url.pathname, method = (init.method || "GET").toUpperCase();
    const body = init.body ? JSON.parse(init.body) : {};
    let m;
    if (path === "/api/config/default") return json(DATA.default_config);
    if (path === "/api/presets") return json(DATA.presets);
    if (path === "/api/catalog") return json({ ...DATA.catalog, gpu: DATA.gpu });
    if (path === "/api/status") return json({ active_run: null, gpu: DATA.gpu });
    if (path === "/api/doctor") return json(DATA.doctor || DOCTOR);
    if (path === "/api/teacher/models") return json({ ok: true, models: DATA.teacher_models });
    if (path === "/api/teacher/check") {
      const model = (body.teacher || {}).model || "qwen3.5:9b";
      const small = /0\.8b|coding_demo/.test(model);
      return json({ ok: true, model, api: "ollama", seconds: small ? 1.6 : 7.1, has_reasoning: false,
        tokens_per_second: small ? 178.7 : 35.9, gpu: { gpu_fraction: 1, vram_gb: small ? 0.6 : 5.49, size_gb: small ? 0.6 : 5.49 },
        models: DATA.teacher_models, reply: "(recorded result from the real run)" });
    }
    if (path === "/api/student/advice") return json(advice(body));
    if (path === "/api/runs" && method === "GET") return json(DATA.runs);
    if ((m = path.match(/^\/api\/runs\/([^/]+)\/samples$/))) {
      const name = decodeURIComponent(m[1]);
      const split = url.searchParams.get("split") || "train";
      const n = parseInt(url.searchParams.get("n") || "5", 10);
      return json(((DATA.samples[name] || {})[split] || []).slice(0, n));
    }
    if ((m = path.match(/^\/api\/runs\/([^/]+)\/process-log$/))) return json({ text: DATA.process_log[decodeURIComponent(m[1])] || "" });
    if ((m = path.match(/^\/api\/runs\/([^/]+)$/)) && method === "GET") {
      const d = DATA.details[decodeURIComponent(m[1])];
      return d ? json(d) : json({ detail: "run not found" }, 404);
    }
    if (path === "/api/quit") return json({ quitting: false });
    return denied(); // start / resume / stop / custom specialties / uploads / previews
  };

  // live log: replay the recorded log like a stream
  class DemoEventSource {
    constructor(url) {
      this.onmessage = null;
      this.closed = false;
      const name = decodeURIComponent((url.match(/\/api\/runs\/([^/]+)\/logs/) || [])[1] || "");
      ready.then(() => {
        const lines = DATA.logs[name] || [];
        let i = 0;
        const tick = () => {
          if (this.closed || i >= lines.length) return;
          const batch = lines.slice(i, i + 3);
          i += batch.length;
          batch.forEach((l) => this.onmessage && this.onmessage({ data: l }));
          setTimeout(tick, 60);
        };
        setTimeout(tick, 150);
      });
    }
    close() { this.closed = true; }
  }
  window.EventSource = DemoEventSource;

  // after the UI loads: hide Quit, show the finished demo run straight away
  window.addEventListener("load", () => {
    const q = document.getElementById("quit-btn");
    if (q) q.style.display = "none";
    ready.then(() => setTimeout(() => {
      if (typeof selectRun === "function" && DATA.runs[0]) selectRun(DATA.runs[0].name);
    }, 900));
  });
  window.DEMO_READ_ONLY_MESSAGE = READ_ONLY;
})();
