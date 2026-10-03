// AgentCreator web UI - plain JS, no build step.
const STAGES = ["generate", "filter", "train", "eval", "export", "benchmark"];
const STAGE_HELP = {
  generate: "teacher answers tasks", filter: "verify & clean", train: "LoRA distillation",
  eval: "score vs base", export: "GGUF file", benchmark: "tokens / second",
};
const SIZES = {
  quick: { data: { max_prompts: 150, samples_per_prompt: 1 }, training: { epochs: 2 }, eval: { max_samples: 50 } },
  standard: { data: { max_prompts: null, samples_per_prompt: 1 }, training: { epochs: 2 }, eval: { max_samples: 100 } },
  thorough: { data: { max_prompts: null, samples_per_prompt: 2 }, training: { epochs: 3 }, eval: { max_samples: 200 } },
};
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : (x * 100).toFixed(1) + "%");

let DEFAULTS = {}, PRESETS = [], CATALOG = { students: [], teacher_suggestions: [] };
let selected = null, logSource = null, detailTimer = null, activeTab = "log";
const W = { student: null, specialty: null, size: "standard", advice: null, nameDirty: false };

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });

// ------------------------------------------------------------- object helpers
const getPath = (o, p) => p.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
function setPath(o, p, v) {
  const ks = p.split("."); let n = o;
  ks.slice(0, -1).forEach((k) => (n = n[k] ??= {}));
  n[ks.at(-1)] = v;
}
function deepMerge(a, b) {
  const out = structuredClone(a);
  for (const [k, v] of Object.entries(b || {})) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) && typeof out[k] === "object" && out[k] ? deepMerge(out[k], v) : v;
  }
  return out;
}

// ------------------------------------------------------------- step 1: teacher
function teacherModel() {
  return $("#t-model").value === "__custom" ? $("#t-model-custom").value.trim() : $("#t-model").value;
}
function teacherCfg() { return { base_url: $("#t-url").value.trim(), model: teacherModel() }; }

async function connectTeacher() {
  const st = $("#t-status");
  st.className = "hint"; st.textContent = "Connecting…";
  const r = await post("/api/teacher/models", { base_url: $("#t-url").value.trim() }).catch((e) => ({ ok: false, models: [], error: e.message }));
  const want = teacherModel() || DEFAULTS.teacher?.model;
  const installed = r.models || [];
  const suggestions = CATALOG.teacher_suggestions.filter((m) => !installed.includes(m));
  $("#t-model").innerHTML =
    (installed.length ? `<optgroup label="Installed on the server">${installed.map((m) => `<option>${esc(m)}</option>`).join("")}</optgroup>` : "") +
    `<optgroup label="Suggestions (pull first)">${suggestions.map((m) => `<option>${esc(m)}</option>`).join("")}</optgroup>` +
    `<option value="__custom">Other…</option>`;
  $("#t-model").value = installed.includes(want) ? want : installed.find((m) => /qwen3\.5:9b/.test(m)) || installed[0] || want || suggestions[0];
  onTeacherChange();
  if (r.ok) {
    st.className = "hint ok";
    st.textContent = installed.length ? `Connected · ${installed.length} model(s) installed` : "Connected, but no models installed - run: ollama pull qwen3.5:9b";
  } else {
    st.className = "hint bad";
    st.textContent = `${r.error || "Cannot reach the server"}\nIs Ollama running? (ollama serve)`;
  }
}

function onTeacherChange() {
  const custom = $("#t-model").value === "__custom";
  $("#t-model-custom").classList.toggle("hidden", !custom);
  const opt = $("#t-model").selectedOptions[0];
  const notInstalled = opt && opt.parentElement?.label?.startsWith("Suggestions");
  if (notInstalled) {
    $("#t-status").className = "hint";
    $("#t-status").textContent = `Not installed yet - run: ollama pull ${teacherModel()}`;
  }
  updateSummary();
}

async function testTeacher() {
  const st = $("#t-status");
  st.className = "hint"; st.textContent = "Asking the teacher a question…";
  try {
    const r = await post("/api/teacher/check", { teacher: teacherCfg() });
    if (r.ok) {
      st.className = "hint ok";
      st.textContent = `✓ ${r.model} answered in ${r.seconds}s` + (r.tokens_per_second ? ` · ${r.tokens_per_second} tokens/s` : "") +
        (r.gpu ? ` · ${Math.round((r.gpu.gpu_fraction || 0) * 100)}% on GPU` : "");
      if (r.warning) { st.className = "hint bad"; st.textContent += "\n⚠ " + r.warning; }
    } else { st.className = "hint bad"; st.textContent = "✗ " + r.error; }
  } catch (e) { st.className = "hint bad"; st.textContent = "✗ " + e.message; }
}

// ------------------------------------------------------------- step 2: student
function renderStudents() {
  const items = CATALOG.students.map((s) => `
    <label class="choice ${W.student === s.id ? "on" : ""}">
      <input type="radio" name="student" value="${esc(s.id)}" ${W.student === s.id ? "checked" : ""}>
      <span class="c-main"><b>${esc(s.id.split("/")[1])}</b> <span class="pill">${s.params}B</span><br><span class="c-note">${esc(s.note)}</span></span>
    </label>`).join("");
  const isCustom = W.student && !CATALOG.students.some((s) => s.id === W.student);
  $("#s-list").innerHTML = items + `
    <label class="choice ${isCustom ? "on" : ""}"><input type="radio" name="student" value="__custom" ${isCustom ? "checked" : ""}>
      <span class="c-main"><b>Custom…</b><br><span class="c-note">any Hugging Face model id or local path</span></span></label>`;
  $("#s-custom").classList.toggle("hidden", !isCustom);
  if (isCustom) $("#s-custom").value = W.student;
  $("#s-list").querySelectorAll("input").forEach((i) => i.addEventListener("change", () => {
    if (i.value === "__custom") { W.student = $("#s-custom").value.trim() || ""; $("#s-custom").classList.remove("hidden"); $("#s-custom").focus(); }
    else W.student = i.value;
    renderStudents(); studentAdvice(); autoName();
  }));
}

let adviceTimer = null;
function studentAdvice() {
  clearTimeout(adviceTimer);
  adviceTimer = setTimeout(async () => {
    if (!W.student) { $("#s-advice").innerHTML = ""; return; }
    const cfg = buildConfig();
    try {
      W.advice = await post("/api/student/advice", {
        model: W.student, load_in_4bit: $("#s-4bit").checked,
        max_seq_length: cfg.student.max_seq_length, batch_size: cfg.training.batch_size,
      });
    } catch { W.advice = null; }
    const a = W.advice;
    $("#s-advice").innerHTML = !a ? "" :
      (a.estimated_vram_gb ? `<span class="pill">≈ ${a.estimated_vram_gb} GB VRAM to train</span> ` : "") +
      a.warnings.map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("");
    updateSummary();
  }, 250);
}

// ------------------------------------------------------------- step 3: specialty
function renderSpecialties() {
  $("#sp-tiles").innerHTML = PRESETS.map((p) => `
    <button type="button" class="tile ${W.specialty === p.name ? "on" : ""}" data-name="${esc(p.name)}">
      <b>${esc(p.title)}</b><small>${p.custom ? "custom" : "built-in"}${p.keep_thinking ? " · thinking" : ""}</small>
      ${p.custom ? `<span class="del" data-del="${esc(p.name)}" title="Delete this specialty">×</span>` : ""}
    </button>`).join("") +
    `<button type="button" class="tile add" id="sp-add"><b>+ Custom</b><small>describe your own</small></button>`;
  $("#sp-tiles").querySelectorAll(".tile[data-name]").forEach((t) => t.addEventListener("click", (e) => {
    if (e.target.dataset.del) return deleteSpecialty(e.target.dataset.del);
    selectSpecialty(t.dataset.name);
  }));
  $("#sp-add").addEventListener("click", () => { $("#custom-form").classList.remove("hidden"); $("#c-title").focus(); });
  const p = PRESETS.find((x) => x.name === W.specialty);
  $("#sp-desc").textContent = p ? `${p.description}\nTraining tasks: ${p.sources.join(", ") || "-"} · Scored on: ${p.eval}` : "";
}

function selectSpecialty(name) {
  W.specialty = name;
  $("#custom-form").classList.add("hidden");
  renderSpecialties(); applyBase(); autoName();
}

async function deleteSpecialty(name) {
  if (!confirm(`Delete the custom specialty "${name}"? Existing runs keep their data.`)) return;
  await api(`/api/specialties/${encodeURIComponent(name)}`, { method: "DELETE" }).catch(() => {});
  PRESETS = await api("/api/presets");
  if (W.specialty === name) W.specialty = PRESETS[0]?.name;
  renderSpecialties(); applyBase();
}

function customSpec() {
  return {
    title: $("#c-title").value.trim(),
    description: $("#c-desc").value.trim(),
    examples: $("#c-examples").value.split("\n").map((s) => s.trim()).filter(Boolean),
    answer_check: $("#c-check").value,
    num_prompts: Number($("#c-num").value || 0),
    keep_thinking: $("#c-think").checked,
  };
}

let uploaded = null;
async function onFileChosen() {
  const f = $("#c-file").files[0];
  const info = $("#c-file-info");
  uploaded = null;
  if (!f) { info.textContent = ""; return; }
  info.className = "hint"; info.textContent = "Reading file…";
  try {
    const content = await f.text();
    uploaded = await post("/api/uploads", { filename: f.name, content });
    info.className = "hint ok";
    info.textContent = `✓ ${uploaded.count} prompts loaded, e.g. "${uploaded.preview[0]}"`;
  } catch (e) { info.className = "hint bad"; info.textContent = "✗ " + e.message; }
}

async function previewCustom() {
  const spec = customSpec(), msg = $("#c-msg");
  if (!spec.title || spec.description.length < 10) { msg.className = "hint bad"; msg.textContent = "Give it a name and a description first."; return; }
  msg.className = "hint"; msg.textContent = `Asking ${teacherModel()} to write a few sample tasks…`;
  $("#c-preview-list").innerHTML = "";
  try {
    const r = await post("/api/specialties/preview", { teacher: teacherCfg(), title: spec.title, description: spec.description, examples: spec.examples });
    msg.className = "hint ok"; msg.textContent = "Sample tasks the teacher would write (each run writes fresh ones):";
    $("#c-preview-list").innerHTML = r.prompts.map((p) => `<li>${esc(p)}</li>`).join("");
  } catch (e) { msg.className = "hint bad"; msg.textContent = "✗ " + e.message; }
}

async function saveCustom() {
  const spec = customSpec(), msg = $("#c-msg");
  if (uploaded) spec.upload = uploaded.path;
  const hf = $("#c-hf").value.trim();
  if (hf) spec.hf_dataset = { path: hf, prompt_field: $("#c-hf-field").value.trim() || "prompt" };
  try {
    const r = await post("/api/specialties", spec);
    PRESETS = await api("/api/presets");
    msg.className = "hint ok"; msg.textContent = `Saved as "${r.name}".`;
    ["#c-title", "#c-desc", "#c-examples", "#c-hf", "#c-hf-field"].forEach((s) => ($(s).value = ""));
    $("#c-file").value = ""; uploaded = null; $("#c-file-info").textContent = ""; $("#c-preview-list").innerHTML = "";
    selectSpecialty(r.name);
  } catch (e) { msg.className = "hint bad"; msg.textContent = "✗ " + e.message; }
}

// ------------------------------------------------------------- step 4: training + config
function baseConfig() {
  const p = PRESETS.find((x) => x.name === W.specialty);
  return deepMerge(deepMerge(DEFAULTS, p?.config_overrides || {}), SIZES[W.size]);
}

function applyBase() {  // refresh the advanced form with the effective defaults
  const cfg = baseConfig();
  for (const el of $("#adv").querySelectorAll("[name]")) {
    const v = getPath(cfg, el.name);
    if (el.type === "checkbox") el.checked = !!v;
    else if (el.name === "data.keep_thinking") el.value = v == null ? "null" : String(v);
    else el.value = v ?? "";
  }
  studentAdvice(); updateSummary();
}

function buildConfig() {
  const cfg = baseConfig();
  for (const el of $("#adv").querySelectorAll("[name]")) {
    let v;
    if (el.type === "checkbox") v = el.checked;
    else if (el.name === "data.keep_thinking") v = el.value === "null" ? null : el.value === "true";
    else if (el.type === "number") v = el.value === "" ? null : Number(el.value);
    else v = el.value.trim() === "" ? null : el.value.trim();
    setPath(cfg, el.name, v);
  }
  cfg.teacher = { ...cfg.teacher, ...teacherCfg() };
  cfg.student = { ...cfg.student, model: W.student, load_in_4bit: $("#s-4bit").checked };
  cfg.specialty = W.specialty;
  cfg.project_name = $("#run-name").value.trim() || autoNameValue();
  return cfg;
}

function autoNameValue() {
  const stu = (W.student || "student").split("/").pop().toLowerCase();
  return `${W.specialty || "custom"}-${stu}`.replace(/[^a-z0-9._-]+/g, "-");
}
function autoName() { if (!W.nameDirty) $("#run-name").value = autoNameValue(); updateSummary(); }

function updateSummary() {
  const p = PRESETS.find((x) => x.name === W.specialty);
  const t = teacherModel() || "?", s = (W.student || "?").split("/").pop();
  const vram = W.advice?.estimated_vram_gb ? ` · ≈ ${W.advice.estimated_vram_gb} GB VRAM` : "";
  $("#summary").innerHTML = `<b>${esc(t)}</b> teaches <b>${esc(s)}</b> to be a <b>${esc(p?.title || "?")}</b> specialist` +
    `<small>${esc(W.size)} run${vram}</small>`;
}

async function startRun() {
  const msg = $("#form-msg");
  const cfg = buildConfig();
  const missing = [!cfg.teacher.model && "teacher model", !cfg.student.model && "student model", !cfg.specialty && "specialty"].filter(Boolean);
  if (missing.length) { msg.className = "hint bad"; msg.textContent = "Choose a " + missing.join(", ") + " first."; return; }
  const stages = [...$("#stages-pick").querySelectorAll("input:checked")].map((i) => i.value);
  const overrides = $("#overrides").value.split("\n").map((l) => l.trim()).filter((l) => l.includes("="));
  $("#start").disabled = true;
  try {
    const r = await post("/api/runs", { config: cfg, stages, overrides });
    msg.className = "hint ok"; msg.textContent = `Started "${r.started}" (${r.stages.join(" → ")})`;
    await loadRuns(); selectRun(r.started);
  } catch (e) { msg.className = "hint bad"; msg.textContent = e.message; }
  finally { $("#start").disabled = false; }
}

// ------------------------------------------------------------- init
async function init() {
  [DEFAULTS, PRESETS, CATALOG] = await Promise.all([api("/api/config/default"), api("/api/presets"), api("/api/catalog")]);
  $("#t-url").value = DEFAULTS.teacher.base_url;
  W.student = DEFAULTS.student.model;
  W.specialty = PRESETS.some((p) => p.name === DEFAULTS.specialty) ? DEFAULTS.specialty : PRESETS[0]?.name;
  // fastest end-to-end defaults: demo specialty, Quick size, the 0.8B student
  if (PRESETS.some((p) => p.name === "coding_demo")) { W.specialty = "coding_demo"; W.size = "quick"; W.student = "Qwen/Qwen3.5-0.8B"; }
  $("#stages-pick").innerHTML = "<span class='opt'>Stages:</span> " +
    STAGES.map((s) => `<label><input type="checkbox" value="${s}" checked> ${s}</label>`).join("");

  renderStudents(); renderSpecialties(); applyBase(); autoName();
  $("#t-model").innerHTML = `<option>${esc(DEFAULTS.teacher.model)}</option>`;
  connectTeacher();

  $("#t-connect").addEventListener("click", connectTeacher);
  $("#t-url").addEventListener("change", connectTeacher);
  $("#t-model").addEventListener("change", onTeacherChange);
  $("#t-model-custom").addEventListener("input", updateSummary);
  $("#t-test").addEventListener("click", testTeacher);
  $("#s-custom").addEventListener("input", (e) => { W.student = e.target.value.trim(); studentAdvice(); autoName(); });
  $("#s-4bit").addEventListener("change", studentAdvice);
  $("#c-file").addEventListener("change", onFileChosen);
  $("#c-preview").addEventListener("click", previewCustom);
  $("#c-save").addEventListener("click", saveCustom);
  $("#c-cancel").addEventListener("click", () => $("#custom-form").classList.add("hidden"));
  $("#size").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    W.size = b.dataset.size;
    $("#size").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    applyBase();
  }));
  $("#adv").addEventListener("change", () => { studentAdvice(); updateSummary(); });
  $("#run-name").addEventListener("input", () => (W.nameDirty = !!$("#run-name").value.trim()));
  $("#start").addEventListener("click", startRun);
  $("#doctor-btn").addEventListener("click", runDoctor);
  $("#quit-btn").addEventListener("click", quitApp);
  $("#refresh").addEventListener("click", loadRuns);
  $("#stop").addEventListener("click", stopRun);
  $("#resume").addEventListener("click", resumeRun);
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

  await loadRuns();
  pollStatus();
  setInterval(pollStatus, 5000);
  setInterval(loadRuns, 4000);
}

async function quitApp() {
  const running = [...document.querySelectorAll("#runs .badge.ok")].length;
  if (!confirm(running ? "A run is still going - quitting will stop it (you can resume it later). Quit AgentCreator?" : "Quit AgentCreator?")) return;
  await fetch("/api/quit", { method: "POST" }).catch(() => {});
  if (logSource) logSource.close();
  document.body.innerHTML = '<div style="padding:40px;font:16px system-ui;color:#e6edf3">AgentCreator has stopped. You can close this tab.<br><br>' +
    '<span style="color:#8b98a8">Start it again any time with AgentCreator.bat or the Desktop shortcut.</span></div>';
}

async function runDoctor() {
  const box = $("#doctor");
  box.classList.remove("hidden");
  box.innerHTML = '<p class="hint">Checking Python, GPU, packages, teacher server and exporters… (takes a few seconds)</p>';
  try {
    const checks = await api("/api/doctor");
    const icon = { ok: "✓", warn: "!", fail: "✗" };
    const fails = checks.filter((c) => c.status === "fail").length;
    box.innerHTML = `<div class="panel-head"><h3>System check · ${fails ? `<span class="bad-t">${fails} problem(s)</span>` : '<span class="ok-t">ready to train</span>'}</h3>
      <button class="ghost small" id="doctor-close">Close</button></div>
      <ul class="checks">${checks.map((c) => `<li class="${c.status}"><span class="ic">${icon[c.status]}</span><b>${esc(c.name)}</b>
        <span class="d">${esc(c.detail)}</span>${c.fix ? `<span class="fix">→ ${esc(c.fix)}</span>` : ""}</li>`).join("")}</ul>`;
    $("#doctor-close").addEventListener("click", () => box.classList.add("hidden"));
  } catch (e) { box.innerHTML = `<p class="hint bad">System check failed: ${esc(e.message)}</p>`; }
}

async function pollStatus() {
  try {
    const s = await api("/api/status");
    $("#gpu").textContent = s.gpu
      ? `${s.gpu.name} · ${(s.gpu.memory_used_mb / 1024).toFixed(1)}/${(s.gpu.memory_total_mb / 1024).toFixed(1)} GB · ${s.gpu.utilization}%`
      : "GPU: nvidia-smi not found";
  } catch { $("#gpu").textContent = "server offline"; }
}

// ------------------------------------------------------------- runs list
async function loadRuns() {
  let runs = [];
  try { runs = await api("/api/runs"); } catch { return; }
  $("#runs").innerHTML = runs.length ? runs.map((r) => `
    <li data-name="${esc(r.name)}" class="${r.name === selected ? "sel" : ""}">
      <div><div class="name">${esc(r.name)} ${r.active ? '<span class="badge ok">running</span>' : ""}</div>
      <div class="meta">${esc(r.specialty || "")} · ${esc((r.student || "").split("/").pop())} ← ${esc(r.teacher || "")}</div></div>
      <div class="dots">${STAGES.map((s) => `<span class="dot ${r.stages[s]?.status}" title="${s}: ${r.stages[s]?.status}"></span>`).join("")}</div>
    </li>`).join("") : '<li class="meta">No runs yet - start one on the left.</li>';
  $("#runs").querySelectorAll("li[data-name]").forEach((li) => li.addEventListener("click", () => selectRun(li.dataset.name)));
}

function selectRun(name) {
  selected = name;
  document.querySelectorAll("#runs li").forEach((li) => li.classList.toggle("sel", li.dataset.name === name));
  $("#detail").classList.remove("hidden");
  $("#log").textContent = "";
  if (logSource) logSource.close();
  logSource = new EventSource(`/api/runs/${encodeURIComponent(name)}/logs`);
  logSource.onmessage = (e) => {
    const el = $("#log"); const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.textContent += e.data + "\n";
    if (atBottom) el.scrollTop = el.scrollHeight;
  };
  clearInterval(detailTimer);
  loadDetail();
  detailTimer = setInterval(loadDetail, 2500);
  if (activeTab !== "log") switchTab(activeTab);
}

let lastDetail = null;
async function loadDetail() {
  if (!selected) return;
  let d;
  try { d = await api(`/api/runs/${encodeURIComponent(selected)}`); } catch { return; }
  lastDetail = d;
  $("#d-title").textContent = d.name;
  $("#d-meta").textContent = `${d.specialty} · teacher ${d.teacher} → student ${d.student}`;
  $("#stop").classList.toggle("hidden", !d.active);
  $("#resume").disabled = d.active;

  const ticked = new Set([...document.querySelectorAll("#pipeline .rs:checked")].map((i) => i.value));
  $("#pipeline").innerHTML = STAGES.map((s) => {
    const st = d.stages[s] || { status: "pending" };
    const p = st.progress; const w = p && p.total ? Math.min(100, (100 * p.done) / p.total) : st.status === "done" ? 100 : 0;
    return `<li class="${st.status}">
      <div class="top"><label class="check" style="margin:0"><input type="checkbox" class="rs" value="${s}" ${ticked.has(s) ? "checked" : ""}> <b>${s}</b></label><span class="st">${st.status}</span></div>
      <div class="bar"><div style="width:${w}%"></div></div>
      <div class="pnote">${p ? `${p.done}/${p.total} ${esc(p.note || "")}` : esc(STAGE_HELP[s])}</div>
      ${st.error ? `<div class="err">${esc(st.error)}</div>` : ""}
    </li>`;
  }).join("");
  renderCards(d);
  renderLoss(d.loss_curve || []);
}

function card(k, v, s = "") { return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }

function renderCards(d) {
  const c = [];
  if (d.data_stats) {
    const ds = d.data_stats;
    c.push(card("Training examples", ds.train_examples, `${ds.raw} teacher answers · ${pct(ds.pass_rate)} passed checks`));
  }
  if (d.train?.metrics) {
    const m = d.train.metrics;
    c.push(card("Final loss", (m.train_loss ?? 0).toFixed(3), m.eval_loss != null ? `validation ${m.eval_loss.toFixed(3)}` : d.train.backend));
  }
  if (d.eval?.models?.distilled) {
    const e = d.eval; const dist = e.models.distilled.score; const base = e.models.base?.score;
    const judged = e.models.distilled.mean_judge_score != null;
    const fmt = (x) => (judged ? `${(x * 10).toFixed(1)}/10` : pct(x));
    const diff = judged ? `${((dist - base) * 10).toFixed(1)}` : `${((dist - base) * 100).toFixed(1)} pts`;
    const gain = base != null ? `<span class="${dist >= base ? "up" : "down"}">${dist >= base ? "+" : ""}${diff}</span> vs untrained ${fmt(base)}` : "";
    c.push(card(judged ? "Teacher-graded score" : `${esc(e.benchmark)} pass rate`, fmt(dist), gain));
  }
  if (d.benchmark?.student) {
    const b = d.benchmark;
    const t = b.teacher?.generate_tokens_per_second;
    c.push(card("Student speed", `${b.student.generate_tokens_per_second} tok/s`,
      t ? `teacher ${t} tok/s · <b>${b.speedup}× faster</b>` : esc(b.student.method)));
  }
  if (d.export?.gguf) {
    c.push(card("GGUF", `${d.export.size_mb} MB`, `${esc(d.export.quantization)}<code>${esc(d.export.ollama)}</code>`));
  } else if (d.export?.ollama_model) {
    c.push(card("In Ollama", esc(d.export.ollama_model), `${esc(d.export.quantization)}<code>${esc(d.export.ollama)}</code>`));
  }
  $("#cards").innerHTML = c.join("");
}

function renderLoss(points) {
  const tr = points.filter((p) => p.loss != null);
  const ev = points.filter((p) => p.eval_loss != null);
  $("#loss-wrap").classList.toggle("hidden", tr.length < 2);
  if (tr.length < 2) return;
  const W = 600, H = 180, pad = 28;
  const maxStep = Math.max(...points.map((p) => p.step));
  const ys = [...tr.map((p) => p.loss), ...ev.map((p) => p.eval_loss)];
  const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
  const x = (s) => pad + ((W - pad - 8) * s) / maxStep;
  const y = (v) => 8 + (H - 8 - 20) * (1 - (v - lo) / span);
  const grid = [0, 0.5, 1].map((f) => { const v = lo + f * span; return `<line class="grid" x1="${pad}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="0" y="${y(v) + 3}">${v.toFixed(2)}</text>`; }).join("");
  const path = tr.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");
  const dots = ev.map((p) => `<circle class="evalpt" cx="${x(p.step)}" cy="${y(p.eval_loss)}" r="4"><title>eval ${p.eval_loss.toFixed(3)}</title></circle>`).join("");
  $("#loss").innerHTML = `${grid}<path class="train" d="${path}"/>${dots}<text x="${W - 70}" y="${H - 4}">step ${maxStep}</text>`;
}

// ------------------------------------------------------------- actions
async function stopRun() {
  if (!selected) return;
  try { await api(`/api/runs/${encodeURIComponent(selected)}/stop`, { method: "POST" }); } catch (e) { alertMsg(e.message); }
  loadDetail(); loadRuns();
}

async function resumeRun() {
  const stages = [...document.querySelectorAll("#pipeline .rs:checked")].map((i) => i.value);
  if (!stages.length) return alertMsg("Tick the stages you want to (re)run in the boxes above.");
  try { await api(`/api/runs/${encodeURIComponent(selected)}/resume`, { method: "POST", body: JSON.stringify({ stages }) }); }
  catch (e) { return alertMsg(e.message); }
  loadDetail(); loadRuns();
}

function alertMsg(m) { $("#d-meta").textContent = m; }

async function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
  ["log", "samples", "evalout", "proc"].forEach((id) => $("#" + id).classList.toggle("hidden", id !== tab));
  if (!selected) return;
  const n = encodeURIComponent(selected);
  if (tab === "samples") {
    const rows = await api(`/api/runs/${n}/samples?n=8&split=train`).catch(() => []);
    $("#samples").innerHTML = rows.length ? rows.map((r) => `<div class="sample"><div class="h"><span>${esc(r.id)}</span><span>${esc(r.source)}</span></div>
      <pre class="user">${esc(r.user)}</pre><pre>${esc(r.assistant)}</pre></div>`).join("") : '<p class="hint">No training data yet (runs after the filter stage).</p>';
  } else if (tab === "evalout") {
    const rows = await api(`/api/runs/${n}/samples?n=10&split=eval`).catch(() => []);
    $("#evalout").innerHTML = rows.length ? rows.map((r) => `<div class="sample"><div class="h"><span>${esc(r.id)}</span>
      <span class="badge ${r.ok ? "ok" : "bad"}">${esc(r.reason)}</span></div><pre>${esc(r.output)}</pre></div>`).join("") : '<p class="hint">No eval outputs yet.</p>';
  } else if (tab === "proc") {
    const r = await api(`/api/runs/${n}/process-log`).catch(() => ({ text: "" }));
    $("#proc").textContent = r.text || "(empty)"; $("#proc").scrollTop = 1e9;
  }
}

init();
