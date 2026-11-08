// Ira TTS in the browser: rumik-oss-1 decoder (2-bit weights, fp16 activations) on WebGPU via onnxruntime-web + Mimi codec decoder.
import * as ort from "./vendor/ort/ort.webgpu.min.mjs";
import { Tokenizer } from "./vendor/tokenizers/tokenizers.mjs";
import * as L from "./lib.js";

const $ = (id) => document.getElementById(id);
// Where the model files live. Default: ./models/ next to this page. Override with ?hf=<user>/<repo> (Hugging Face) or ?models=<url-prefix>.
const Q = new URLSearchParams(location.search);
const HF_DEFAULT = "Sanyam0605/ira-tts-2bit-web";
let BASE = Q.get("models") ? Q.get("models").replace(/\/?$/, "/") : Q.get("hf") ? `https://huggingface.co/${Q.get("hf")}/resolve/main/` : "models/";
// Default: use ./models/ if it exists next to the page, otherwise the public Hugging Face repo.
async function resolveBase() {
  if (Q.get("models") || Q.get("hf")) return;
  try { const r = await fetch("models/assets/web_vocab.json", { method: "HEAD" }); if (r.ok) return; } catch (e) { /* fall through */ }
  BASE = `https://huggingface.co/${HF_DEFAULT}/resolve/main/`;
}
const NL = 36, NKV = 4, HD = 128, MAXP = 4096; // MAXP: static KV slots of the fused-GQA graph
const PROFILE = Q.get("profile") === "1";
const STREAM = Q.get("stream") === "1"; // opt-in: play the first 25 frames while the rest of the sentence is still generating
const FIRST_CHUNK_TOKENS = 200; // 25 frames * 8 tokens
const CTX_FRAMES = 16, SAMPLES_PER_FRAME = 1920; // codec left context for the tail chunk; 24 kHz / 12.5 fps
const prof = [];
const state = { gqa: false, batched: false, dec: null, mimi: null, tokenizer: null, vocab: null, ready: false, busy: false, cancel: false, ctx: null, nextStart: 0, wavParts: [] };

function log(...a) {
  const s = a.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join(" ");
  const el = $("log"); el.textContent += `[${((performance.now() - T0) / 1000).toFixed(1)}s] ${s}\n`; el.scrollTop = el.scrollHeight; console.log(s);
}
const T0 = performance.now();
const setBar = (f, msg) => { $("loadBar").style.width = `${Math.round(f * 100)}%`; if (msg !== undefined) $("loadMsg").textContent = msg; };
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "–");

// ---------------------------------------------------------------- GPU info
async function gpuInfo() {
  if (!navigator.gpu) { $("gpuInfo").innerHTML = '<span class="bad">WebGPU not available in this browser</span>'; return null; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { $("gpuInfo").innerHTML = '<span class="bad">no WebGPU adapter</span>'; return null; }
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  const f16 = adapter.features.has("shader-f16");
  const desc = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" · ") || "unknown GPU";
  $("gpuInfo").innerHTML = `${desc} · fp16 ${f16 ? '<span class="ok">yes</span>' : '<span class="bad">NO</span>'} · maxBuffer ${(adapter.limits.maxBufferSize / 2 ** 30).toFixed(1)} GiB`;
  log("adapter:", { ...info, shaderF16: f16, maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize });
  return { adapter, info, f16 };
}

// ---------------------------------------------------------------- fetching with progress + Cache Storage
async function fetchBytes(url, label, onProgress) {
  const key = new URL(url, location.href).href;
  try {
    const cache = await caches.open("ira-tts-v1");
    const hit = await cache.match(key);
    if (hit) { log(`${label}: from cache`); const b = new Uint8Array(await hit.arrayBuffer()); onProgress?.(1); return b; }
  } catch (e) { /* cache unavailable */ }
  const resp = await fetch(key);
  if (!resp.ok) throw new Error(`${label}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  let buf = total ? new Uint8Array(total) : null, got = 0, chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf) buf.set(value, got); else chunks.push(value);
    got += value.length; onProgress?.(total ? got / total : 0, got);
  }
  if (!buf) { buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; } }
  try { const cache = await caches.open("ira-tts-v1"); await cache.put(key, new Response(buf)); } catch (e) { log(`${label}: not cached (${e.message})`); }
  return buf;
}

// ---------------------------------------------------------------- model loading
async function loadModels() {
  const gi = await gpuInfo();
  if (!gi) throw new Error("WebGPU is required");
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.webgpu.powerPreference = "high-performance";
  if (PROFILE) { ort.env.webgpu.profiling = { mode: "default", ondata: (d) => prof.push(d) }; log("GPU profiling ON (timestamp queries)"); }
  $("loadBtn").disabled = true;
  const t0 = performance.now();
  await resolveBase();

  setBar(0.01, "tokenizer…");
  const [tj, tc, wv] = await Promise.all(["assets/tokenizer.json", "assets/tokenizer_config.json", "assets/web_vocab.json"].map((u) => fetch(BASE + u).then((r) => { if (!r.ok) throw new Error(`${BASE + u}: HTTP ${r.status}`); return r.json(); })));
  state.tokenizer = new Tokenizer(tj, tc); state.vocab = L.buildVocab(wv);
  log(`model files from: ${new URL(BASE, location.href).href}`); log(`tokenizer ready (${state.vocab.nVocab} embedding rows)`);

  setBar(0.03, "downloading decoder weights…");
  const weights = await fetchBytes(BASE + "decoder/model.onnx.data", "decoder weights", (f, got) => setBar(0.03 + 0.6 * f, `decoder weights ${(got / 1e6).toFixed(0)} MB`));
  const graph = await fetchBytes(BASE + "decoder/model.onnx", "decoder graph");
  setBar(0.65, "creating decoder session (compiling WebGPU shaders)…");
  const presentLoc = {};
  for (let i = 0; i < NL; i++) { presentLoc[`present.${i}.key`] = "gpu-buffer"; presentLoc[`present.${i}.value`] = "gpu-buffer"; }
  const ts = performance.now();
  state.dec = await ort.InferenceSession.create(graph, {
    executionProviders: ["webgpu"], externalData: [{ path: "model.onnx.data", data: weights }],
    preferredOutputLocation: presentLoc, graphOptimizationLevel: "all",
  });
  state.gqa = state.dec.inputNames.includes("seqlens_k"); // fused GroupQueryAttention graph: one token per run, static KV
  const md = state.dec.inputMetadata, idMeta = Array.isArray(md) ? md.find((m) => m.name === "input_ids") : md?.input_ids;
  log("input_ids metadata:", idMeta);
  state.batched = state.gqa && !!idMeta?.shape && idMeta.shape[1] !== 1; // symbolic T: batched prefill
  log(`decoder graph: ${state.gqa ? `fused GQA, static KV, ${state.batched ? "batched prefill" : "token-by-token prefill"}` : "shipped traced graph"}`);
  log(`decoder session created in ${((performance.now() - ts) / 1000).toFixed(1)} s`);
  try { const a = ort.env.webgpu.adapter; const inf = a?.info || {}; log("onnxruntime adapter:", { vendor: inf.vendor, architecture: inf.architecture, description: inf.description }); } catch (e) { /* ignore */ }

  setBar(0.85, "downloading Mimi codec…");
  const mbytes = await fetchBytes(BASE + "mimi/decoder_8cb_fp32.onnx", "mimi decoder", (f) => setBar(0.85 + 0.1 * f));
  setBar(0.96, "creating codec session…");
  state.mimi = await ort.InferenceSession.create(mbytes, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
  setBar(1, `ready in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
  state.ready = true;
  $("speakBtn").disabled = false; $("benchBtn").disabled = false;
  log("models ready");
}

// ---------------------------------------------------------------- decoding
const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, (x) => BigInt(x)), dims);
let zeroKV = null;
function emptyPast() {
  const p = {};
  if (state.gqa) {
    zeroKV ??= new Uint16Array(NKV * MAXP * HD); // one shared zero buffer, read-only
    for (let i = 0; i < NL; i++) for (const kv of ["key", "value"]) p[`past_key_values.${i}.${kv}`] = new ort.Tensor("float16", zeroKV, [1, NKV, MAXP, HD]);
    return p;
  }
  for (let i = 0; i < NL; i++) for (const kv of ["key", "value"]) p[`past_key_values.${i}.${kv}`] = new ort.Tensor("float16", new Uint16Array(0), [1, NKV, 0, HD]);
  return p;
}
function disposePast(past) { for (const k in past) { const t = past[k]; if (t.location === "gpu-buffer") { try { t.dispose(); } catch (e) { /* already */ } } } }

/** Run the decoder on `ids` starting at position pos0. The fused GQA graph is single-token, so prefill loops token by token.
 *  Returns {out (outputs of the last run), past}. */
async function decodeStep(ids, pos0, past) {
  if (!state.gqa) {
    const out = await state.dec.run({ input_ids: i64(ids, [1, ids.length]), position_ids: i64(Array.from({ length: ids.length }, (_, k) => pos0 + k), [1, ids.length]), ...past });
    disposePast(past);
    return { out, past: presentToPast(out) };
  }
  if (pos0 + ids.length > MAXP) throw new Error(`context ${pos0 + ids.length} exceeds the ${MAXP}-slot KV buffer`);
  let out;
  if (ids.length > 1 && state.batched) { // one run for the whole prompt (needs the symbolic-T graph, empty past)
    out = await state.dec.run({ input_ids: i64(ids, [1, ids.length]), seqlens_k: new ort.Tensor("int32", Int32Array.of(pos0 + ids.length - 1), [1]), total_sequence_length: new ort.Tensor("int32", Int32Array.of(pos0 + ids.length), []), ...past });
    disposePast(past);
    return { out, past: presentToPast(out) };
  }
  for (let k = 0; k < ids.length; k++) {
    const p = pos0 + k;
    out = await state.dec.run({ input_ids: i64([ids[k]], [1, 1]), seqlens_k: new ort.Tensor("int32", Int32Array.of(p), [1]), total_sequence_length: new ort.Tensor("int32", Int32Array.of(p + 1), []), ...past });
    disposePast(past);
    past = presentToPast(out);
  }
  return { out, past };
}
function presentToPast(out) {
  const past = {};
  for (let i = 0; i < NL; i++) { past[`past_key_values.${i}.key`] = out[`present.${i}.key`]; past[`past_key_values.${i}.value`] = out[`present.${i}.value`]; }
  return past;
}

/** Autoregressively generate audio tokens for one prompt. Returns {tokens (a-values, multiple of 8), stats}. */
async function generate(ids, opts) {
  const T = ids.length;
  const maxSteps = Math.min(Math.round(opts.maxSeconds * L.TOKENS_PER_SECOND), state.gqa ? MAXP - T - 1 : Infinity);
  const rand = opts.seed ? L.rng(opts.seed) : Math.random;
  let past = emptyPast(), inputIds = ids, pos0 = 0;
  const tokens = [];
  const t0 = performance.now();
  let tFirstToken = null, stopReason = "max-length", tPrefill = 0;
  for (let step = 0; ; step++) {
    if (state.cancel) { stopReason = "cancelled"; break; }
    const r = await decodeStep(inputIds, pos0, past);
    const out = r.out; past = r.past;
    const logits = L.toFloat32(await out.logits.getData());
    const stopP = (await out.stop.getData())[0];
    if (step === 0) { tPrefill = performance.now() - t0; }
    const n = tokens.length;
    if (n >= 8 && n % 8 === 0 && stopP > opts.stopTh) { stopReason = "stop-head"; break; }
    if (n >= maxSteps) { stopReason = "max-length"; break; }
    const o = L.sampleToken(logits, n, { temperature: opts.temperature, topK: opts.topK, allowEnd: n >= 8, rand });
    if (o === 0) { stopReason = "end-token"; break; }
    tokens.push(o - 1);
    if (opts.onChunk && tokens.length === FIRST_CHUNK_TOKENS && !state.cancel) await opts.onChunk(tokens.slice()); // awaited: onnxruntime-web must not run two sessions at once
    if (tFirstToken === null) tFirstToken = performance.now() - t0;
    inputIds = [L.outToInputId(o, state.vocab.audioBase)]; pos0 = T + tokens.length - 1;
    if (tokens.length % 40 === 0) opts.onProgress?.(tokens.length, performance.now() - t0);
  }
  disposePast(past);
  const dt = performance.now() - t0;
  const usable = Math.floor(tokens.length / 8) * 8;
  return { tokens: tokens.slice(0, usable), stats: { promptTokens: T, tokens: tokens.length, ms: dt, prefillMs: tPrefill, tps: tokens.length / (dt / 1000), stopReason } };
}

async function decodeAudio(tokens) {
  const { codes, T } = L.mimiCodes(tokens);
  const out = await state.mimi.run({ audio_codes: new ort.Tensor("int64", codes, [1, 32, T]) });
  return out.audio_values.data;
}

// ---------------------------------------------------------------- playback
function audioCtx() { if (!state.ctx || state.ctx.state === "closed") { state.ctx = new AudioContext({ sampleRate: L.SAMPLE_RATE }); state.nextStart = 0; } return state.ctx; }
function enqueue(samples) {
  const ctx = audioCtx(); ctx.resume?.();
  const buf = ctx.createBuffer(1, samples.length, L.SAMPLE_RATE); buf.copyToChannel(samples, 0);
  const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination);
  const t = Math.max(ctx.currentTime + 0.05, state.nextStart); src.start(t); state.nextStart = t + buf.duration;
}

function description() { return [$("tone").value, $("accent").value, $("pace").value].filter(Boolean).length === 3 ? [$("tone").value, $("accent").value, $("pace").value].join(", ") : [$("tone").value, $("accent").value, $("pace").value].filter(Boolean).join(", "); }
function settings() { return { temperature: +$("temp").value || 0.8, topK: +$("topk").value || 30, stopTh: +$("stopTh").value || 0.5, seed: +$("seed").value || 0 }; }

/** Speak a whole text: split into sentences, synthesise each, queue playback gaplessly. Returns summary stats. */
async function speak(text, extra = {}) {
  if (state.busy) return null;
  state.busy = true; state.cancel = false; $("speakBtn").disabled = true; $("stopBtn").disabled = false; $("dl").style.display = "none";
  const set = { ...settings(), ...extra };
  const desc = description();
  const sentences = L.splitSentences(text);
  const tStart = performance.now();
  let totalTokens = 0, totalGenMs = 0, audioSecs = 0, tFirstAudio = null;
  state.wavParts = []; if (state.ctx) { try { await state.ctx.close(); } catch (e) { /* */ } } state.ctx = null;
  const perSentence = [];
  try {
    for (let i = 0; i < sentences.length && !state.cancel; i++) {
      const s = sentences[i];
      const enc = state.tokenizer.encode(L.buildPrompt(s, desc));
      const { ids, oov } = L.toPrunedIds(enc, state.vocab);
      log(`sentence ${i + 1}/${sentences.length}: "${s.slice(0, 60)}" → ${ids.length} prompt tokens${oov ? ` (${oov} rare tokens spelled as bytes)` : ""}`);
      const maxSeconds = Math.min(L.maxSecondsFor(s), 30);
      // STREAM: decode and enqueue the first chunk as soon as it exists, while generation continues.
      // The codec is not streaming, so the tail decoded separately may show a small seam at the chunk boundary.
      let head = null;
      const onChunk = STREAM ? (chunk) => {
        return (head = decodeAudio(chunk).then((pcm) => {
          if (tFirstAudio === null) tFirstAudio = performance.now() - tStart;
          if (!state.cancel) enqueue(Float32Array.from(pcm));
          return pcm;
        }));
      } : null;
      const { tokens, stats } = await generate(ids, { ...set, maxSeconds, onChunk, onProgress: (n, ms) => { $("sTps").textContent = fmt(n / (ms / 1000)); } });
      totalTokens += stats.tokens; totalGenMs += stats.ms;
      if (tokens.length < 8) { log("  (no audio produced)", stats); continue; }
      const tD = performance.now();
      const pieces = [];
      if (head) pieces.push(await head);
      if (head ? tokens.length > FIRST_CHUNK_TOKENS : true) {
        // the tail is decoded with a few frames of left context (dropped from the output) so it matches a whole-sentence decode
        const ctx = head ? CTX_FRAMES * 8 : 0;
        let pcm = await decodeAudio(tokens.slice(head ? FIRST_CHUNK_TOKENS - ctx : 0));
        if (head) pcm = pcm.slice(CTX_FRAMES * SAMPLES_PER_FRAME);
        pieces.push(pcm); enqueue(Float32Array.from(pcm));
      }
      let secs = 0;
      for (const p of pieces) { secs += p.length / L.SAMPLE_RATE; state.wavParts.push(new Float32Array(p)); }
      state.wavParts.push(new Float32Array(Math.round(0.12 * L.SAMPLE_RATE)));
      audioSecs += secs;
      if (tFirstAudio === null) tFirstAudio = performance.now() - tStart;
      perSentence.push({ sentence: s, secs, ...stats, codecMs: performance.now() - tD });
      log(`  → ${secs.toFixed(1)} s audio, ${stats.tokens} tokens in ${(stats.ms / 1000).toFixed(1)} s (${stats.tps.toFixed(1)} tok/s), prefill ${stats.prefillMs.toFixed(0)} ms, codec ${(performance.now() - tD).toFixed(0)} ms, stop: ${stats.stopReason}`);
    }
  } catch (e) { log("ERROR:", String(e && e.stack || e)); throw e; }
  finally { state.busy = false; $("speakBtn").disabled = !state.ready; $("stopBtn").disabled = true; }
  const tps = totalTokens / (totalGenMs / 1000);
  $("sTps").textContent = fmt(tps); $("sRtf").textContent = fmt(((performance.now() - tStart) / 1000) / Math.max(audioSecs, 1e-6), 2);
  $("sFirst").textContent = tFirstAudio ? `${(tFirstAudio / 1000).toFixed(1)} s` : "–"; $("sAudio").textContent = `${audioSecs.toFixed(1)} s`;
  if (state.wavParts.length) {
    const total = state.wavParts.reduce((a, b) => a + b.length, 0), all = new Float32Array(total); let o = 0; for (const p of state.wavParts) { all.set(p, o); o += p.length; }
    const url = URL.createObjectURL(new Blob([L.encodeWav(all)], { type: "audio/wav" })); const a = $("dl"); a.href = url; a.style.display = "inline";
  }
  return { tps, audioSecs, tFirstAudio, totalTokens, perSentence, wallMs: performance.now() - tStart };
}

/** Raw decode speed: short prompt then N forced-valid steps (no sampling effects, no stop). Reports where the time goes. */
async function speedTest(steps = 60) {
  if (state.busy) return null;
  state.busy = true; $("benchBtn").disabled = true; $("speakBtn").disabled = true;
  try {
    const enc = state.tokenizer.encode(L.buildPrompt("Hello, this is a speed test of the voice model.", ""));
    const { ids } = L.toPrunedIds(enc, state.vocab);
    let past = emptyPast(), input = ids, pos0 = 0; const lat = [], runMs = [], dlMs = [], feedMs = [];
    for (let s = 0; s < steps + 1; s++) {
      if (s === 1) prof.length = 0; // profile only the decode steps
      const t = performance.now();
      const t1 = performance.now();
      const r = await decodeStep(input, pos0, past);
      const out = r.out; past = r.past;
      const t2 = performance.now();
      await out.logits.getData();
      const t3 = performance.now();
      if (s === 0) log(`speed test: prefill of ${input.length} tokens took ${(t3 - t).toFixed(0)} ms`); else { lat.push(t3 - t); feedMs.push(t1 - t); runMs.push(t2 - t1); dlMs.push(t3 - t2); }
      input = [state.vocab.audioBase + ((s * 7919) % 2048) * 8 + (s % 8)]; pos0 = ids.length + s;
    }
    disposePast(past);
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const sorted = lat.slice().sort((a, b) => a - b), med = sorted[sorted.length >> 1], mean = avg(lat);
    const res = { steps, medianMs: med, meanMs: mean, p90Ms: sorted[Math.floor(sorted.length * 0.9)], tpsMedian: 1000 / med, tpsMean: 1000 / mean };
    $("sTps").textContent = fmt(res.tpsMean); $("sRtf").textContent = fmt(100 / res.tpsMean, 2); $("sFirst").textContent = "–"; $("sAudio").textContent = "–";
    log(`speed test: ${steps} decode steps · median ${med.toFixed(1)} ms · mean ${mean.toFixed(1)} ms · ${res.tpsMean.toFixed(1)} tok/s → real-time factor ${(100 / res.tpsMean).toFixed(2)} (needs ≥ 100 tok/s for 1.0)`);
    log(`  per step: build feeds ${avg(feedMs).toFixed(1)} ms · session.run ${avg(runMs).toFixed(1)} ms · read logits ${avg(dlMs).toFixed(1)} ms`);
    if (PROFILE) {
      await new Promise((r) => setTimeout(r, 500));
      if (!prof.length) log("  profiling: no GPU timing data (timestamp-query probably unsupported) — send the lines above anyway");
      else {
        const by = new Map(); let total = 0;
        for (const d of prof) { const ms = (Number(d.endTime) - Number(d.startTime)) / 1e6; total += ms; const k = `${d.kernelType} / ${d.programName}`; const e = by.get(k) || { ms: 0, n: 0 }; e.ms += ms; e.n++; by.set(k, e); }
        const rows = [...by.entries()].sort((a, b) => b[1].ms - a[1].ms).slice(0, 14);
        log(`  GPU kernel time per step: ${(total / steps).toFixed(1)} ms of ${mean.toFixed(1)} ms wall · ${(prof.length / steps).toFixed(0)} GPU dispatches per step`);
        for (const [k, e] of rows) log(`    ${(e.ms / steps).toFixed(2).padStart(7)} ms/step  ${(e.n / steps).toFixed(0).padStart(4)}x  ${k}`);
      }
    }
    return res;
  } finally { state.busy = false; $("benchBtn").disabled = false; $("speakBtn").disabled = false; }
}

// ---------------------------------------------------------------- UI wiring
$("loadBtn").onclick = () => loadModels().catch((e) => { log("LOAD ERROR:", String(e && e.stack || e)); $("loadMsg").innerHTML = `<span class="bad">${String(e.message || e)}</span>`; $("loadBtn").disabled = false; });
$("speakBtn").onclick = () => speak($("text").value.trim()).catch((e) => log("ERROR:", String(e)));
$("stopBtn").onclick = () => { state.cancel = true; if (state.ctx) state.ctx.close().catch(() => {}); };
$("benchBtn").onclick = () => speedTest().catch((e) => log("ERROR:", String(e)));
document.querySelectorAll("[data-ex]").forEach((b) => (b.onclick = () => { $("text").value = b.dataset.ex; }));
document.querySelectorAll("[data-tag]").forEach((b) => (b.onclick = () => { const t = $("text"), s = t.selectionStart; t.value = t.value.slice(0, s) + b.dataset.tag + " " + t.value.slice(t.selectionEnd); t.focus(); }));
gpuInfo().catch((e) => log("gpu info error:", String(e)));

// test hook for automated runs (headless browser): window.iraTest.run(text)
window.iraTest = {
  load: loadModels,
  speak: async (text, extra) => { const r = await speak(text, extra); return r; },
  bench: speedTest,
  pcm: () => { const total = state.wavParts.reduce((a, b) => a + b.length, 0), all = new Float32Array(total); let o = 0; for (const p of state.wavParts) { all.set(p, o); o += p.length; } return all; },
  log: () => $("log").textContent,
};
