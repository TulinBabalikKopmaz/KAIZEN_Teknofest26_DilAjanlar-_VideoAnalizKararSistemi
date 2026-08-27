const FORKLIFT = `<svg width="26" height="20" viewBox="0 0 48 40" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">
  <rect x="6" y="14" width="22" height="11" rx="2"></rect>
  <rect x="8" y="5" width="10" height="9" rx="1"></rect>
  <circle cx="12" cy="29" r="5"></circle>
  <circle cx="25" cy="29" r="5"></circle>
  <line x1="31" y1="6" x2="31" y2="29"></line>
  <line x1="31" y1="22" x2="41" y2="22"></line>
  <line x1="31" y1="27" x2="41" y2="27"></line>
</svg>`;

const LAW = {
  ok: { kicker: "İSG mevzuat desteği · 2 madde", articles: [
    { title: "İSG Kanunu 6331 · Md. 22", text: "Sıcak iş / proses ateşi bulunan alanlarda yangın söndürme ekipmanına erişim sürekli açık tutulur." },
    { title: "Kaynak İşleri Yönetmeliği", text: "Kaynak istasyonunda KKD kullanımı ve havalandırma her vardiyada kontrol edilir." }
  ]},
  watch: { kicker: "İSG mevzuat desteği · 2 madde", articles: [
    { title: "İSG Kanunu 6331 · Md. 4", text: "İşveren, araç ve yaya güzergâhlarını fiziksel olarak ayırmakla yükümlüdür." },
    { title: "Ramak Kala Bildirim Prosedürü", text: "Ramak kala olayları 24 saat içinde kayda geçirilir ve saha turu ile doğrulanır." }
  ]},
  critical: { kicker: "İSG mevzuat desteği · 2 madde", articles: [
    { title: "İSG Kanunu 6331 · Md. 26", text: "İş kazası halinde alan derhal güvenlik altına alınır, sağlık ekibi ve yetkili merciler bilgilendirilir." },
    { title: "Kaza Bildirim Yükümlülüğü", text: "İş kazası en geç 3 iş günü içinde SGK'ya bildirilir; kanıt kareleri arşivlenir." }
  ]}
};

const S = {
  theme: localStorage.getItem("kz-theme") || "light",
  view: "demo",
  sidebar: true,
  lock: true,
  provider: "teknofest",
  fast: false,
  rag: true,
  frames: 8,
  phase: "idle",
  doneSteps: 0,
  file: null,
  result: null,
  backup: false,
  jsonOpen: false,
  timingsOpen: false,
  evidenceOpen: false,
  lawOpen: false,
  liveKind: "dosya",
  liveFile: null,
  liveSnap: null,
  countP: 1,
  skipSecond: false,
  holdOverlay: false,
  waitingResult: false
};

let fileBlob = null;
let liveFileBlob = null;
let timers = [];
let livePoll = 0;
let liveFp = "";
let livePreviewUrl = "";

function $(id) { return document.getElementById(id); }
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}
function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function clearTimers() { timers.forEach(clearTimeout); timers = []; }

function sourceInfo() {
  if (S.backup) return { label: "Kayıtlı yedek", tone: "watch", detail: "canlı EVREN sonucu değil" };
  if (S.provider === "ollama") return { label: "Ollama", tone: "critical", detail: "yerel yedek — sunum kalitesi değil" };
  if (S.provider === "mock") return { label: "mock", tone: "watch", detail: "modelsiz deneme" };
  return { label: "EVREN", tone: "ok", detail: "resmi API — sunum" };
}

function setTheme(theme) {
  S.theme = theme;
  localStorage.setItem("kz-theme", theme);
  $("app").setAttribute("data-theme", theme);
  document.querySelectorAll("[data-act=theme]").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.val === theme);
  });
}

function paintChrome() {
  $("sidebar").classList.toggle("is-off", !S.sidebar);
  $("rail").classList.toggle("is-off", S.sidebar);
  document.querySelectorAll("[data-act=view]").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.val === S.view);
  });
  $("demo-view").classList.toggle("is-off", S.view !== "demo");
  $("live-view").classList.toggle("is-off", S.view !== "live");
  const lockBtn = document.querySelector("[data-act=lock]");
  lockBtn.classList.toggle("on", S.lock);
  $("lock-cap").textContent = S.lock
    ? "Kaynak kilitli: EVREN · vlm + llm-fast — Ollama'ya düşülmez"
    : "Sunum kilidi kapalı: EVREN düşerse Ollama denenebilir";
  $("lock-cap").className = "kz-help " + (S.lock ? "ok" : "bad");
  $("provider-box").classList.toggle("is-off", S.lock);
  document.querySelectorAll("[data-act=provider]").forEach((btn) => {
    btn.classList.toggle("on", btn.dataset.val === S.provider);
  });
  document.querySelector("[data-act=fast]").classList.toggle("on", S.fast);
  document.querySelector("[data-act=rag]").classList.toggle("on", S.rag && !S.fast);
  $("frames-val").textContent = String(S.frames);
  $("describe").textContent =
    "sağlayıcı=" + (S.lock ? "teknofest" : S.provider) +
    " · vlm=evren/vlm · llm=evren/llm-fast · hızlı mod=" + (S.fast ? "açık" : "kapalı");
  const running = S.phase === "running" || S.waitingResult;
  $("run").disabled = running;
  $("run").textContent = running ? "Analiz çalışıyor..." : "Analiz et";
}

function stepMs(index) {
  return [820, 1080, S.skipSecond && index === 2 ? 420 : 900, 760, 1020][index];
}

function stepDefs() {
  const t = (S.result && S.result.timings) || {};
  const skip = S.skipSecond;
  return [
    { label: "Kare çıkarma ve sensör kanıtı", detail: S.frames + " kare hedefi", secs: t.kare_ve_kanit },
    { label: "Görsel model analizi", detail: "etiket + olaylar", secs: t.vlm },
    skip
      ? { label: "İkinci bakış", detail: "kanıt sakin, atlandı", secs: 0, skipped: true }
      : { label: "İkinci bakış", detail: "sensör şüpheli", secs: t.ikinci_bakis || 0 },
    { label: "Kural ve birleştirme", detail: "zaman hizalama", secs: t.kural_katmani },
    { label: "Cevap ve saha aksiyonları", detail: S.rag && !S.fast ? "LLM · RAG açık" : "LLM · RAG kapalı", secs: t.cevap }
  ];
}

function renderOverlay() {
  const overlay = $("overlay");
  const on = S.phase === "running" || S.phase === "handoff";
  overlay.classList.toggle("is-off", !on);
  overlay.classList.remove("leaving");
  if (!on) {
    overlay.innerHTML = "";
    return;
  }
  const defs = stepDefs();
  const done = S.doneSteps;
  const rings = defs.map((step, index) => {
    let status = "pending";
    if (S.phase === "running") status = index < done ? "done" : index === done ? "active" : "pending";
    else status = "done";
    const offset = status === "done" ? "0" : "114";
    const color = status === "active" ? "var(--kz-gold)" : status === "done" ? (step.skipped ? "var(--kz-knob-off)" : "var(--kz-ok)") : "rgba(var(--kz-line-rgb),0.1)";
    const center = status === "active" ? FORKLIFT : (status === "done" && !step.skipped ? "✓" : status === "done" ? "–" : String(index + 1));
    const time = status === "done" && step.secs ? Number(step.secs).toFixed(1) + " sn" : status === "active" ? "çalışıyor" : (step.skipped && status === "done" ? "atlandı" : "");
    return `<div class="kz-ring-cell ${status}">
      <div class="kz-line"></div>
      <div class="kz-ring">
        <div class="kz-halo"></div>
        <svg width="52" height="52" viewBox="0 0 52 52">
          <circle cx="26" cy="26" r="18" fill="none" stroke="rgba(var(--kz-line-rgb),0.1)" stroke-width="2"></circle>
          <circle cx="26" cy="26" r="18" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-dasharray="114" stroke-dashoffset="${status === "active" ? 0 : offset}" style="${status === "active" ? "animation:kzArc " + (stepMs(index) / 1000).toFixed(2) + "s linear both" : ""}"></circle>
        </svg>
        <div class="kz-center">${center}</div>
      </div>
    </div>`;
  }).join("");
  const labels = defs.map((step) => `<div class="kz-step-lab">${esc(step.label)}</div>`).join("");
  const details = defs.map((step, index) => {
    const status = S.phase === "running" ? (index < done ? "done" : index === done ? "active" : "pending") : "done";
    return `<div class="kz-step-det">${esc(status === "active" ? step.detail : step.detail)}</div>`;
  }).join("");
  const times = defs.map((step, index) => {
    const status = S.phase === "running" ? (index < done ? "done" : index === done ? "active" : "pending") : "done";
    const time = status === "done" && step.secs ? Number(step.secs).toFixed(1) + " sn" : status === "active" ? "çalışıyor" : "";
    return `<div class="kz-step-time">${esc(time)}</div>`;
  }).join("");
  const statusLine = S.holdOverlay
    ? "Model yanıtı bekleniyor…"
    : S.phase === "running"
      ? "Analiz çalışıyor... (" + done + "/5)"
      : "Tamamlandı (" + ((S.result && S.result.total_s) || 0).toFixed(1) + " sn)";
  overlay.innerHTML = `<section class="kz-flow ${S.phase === "handoff" ? "handoff" : ""}">
    ${S.phase === "running" ? '<div class="kz-sweep"><span></span></div>' : ""}
    <div class="kz-flow-top">
      <div class="kz-kicker-sm">Analiz akışı</div>
      <div class="kz-flow-status">${esc(statusLine)}</div>
    </div>
    <div class="kz-steps">${rings}${labels}${details}${times}</div>
    ${S.phase === "handoff" ? '<div class="kz-flare"></div>' : ""}
  </section>`;
}

function renderResult() {
  const host = $("result");
  const d = S.result;
  if (!d || S.phase !== "done") {
    host.innerHTML = "";
    return;
  }
  const src = d.source || sourceInfo();
  const words = (d.situation + " · " + d.decision).split(" ").map((word, i) =>
    `<span style="animation-delay:${(0.12 + i * 0.11).toFixed(2)}s">${esc(word)}</span>`
  ).join("");
  const p = S.countP;
  const events = d.events || [];
  const actions = d.actions || [];
  const frames = d.frames || [];
  const law = d.law || LAW[d.tone] || LAW.watch;
  const hard = d.hard;
  const frameHtml = frames.length
    ? frames.map((frame) => {
        const time = typeof frame === "string" ? frame : (frame.time || "");
        const url = typeof frame === "string" ? "" : (frame.url || "");
        return `<figure class="kz-shot">${url ? `<img src="${esc(url)}" alt="">` : `<div class="kz-shot-empty"></div>`}<figcaption>${esc(time)}</figcaption></figure>`;
      }).join("")
    : "";
  host.innerHTML = `<section class="kz-result">
    <header class="kz-result-h">
      <div><span class="k">Analiz sonucu</span>
      <span class="m" style="margin-left:16px">${esc(d.video_name || "kayıt")} · Tamamlandı (${(d.total_s * p).toFixed(1)} sn)</span></div>
    </header>
    <div class="kz-result-b">
      ${d.backup ? '<div class="kz-banner">Ekranda <strong>kayıtlı sahne yedeği</strong> var (canlı API sonucu değil). Jüri videosunda Analiz Et ile taze koşu alın.</div>' : ""}
      <div class="kz-verdict ${esc(d.tone)}">
        <div class="kz-vglow"></div>
        <div class="kz-source ${esc(src.tone)}">Kaynak · ${esc(src.label)}<span> ${esc(src.detail || "")}</span></div>
        <div class="kz-vkick">${esc(d.kicker || "Saha kararı")}</div>
        <div class="kz-vtitle">${words}</div>
        <p class="kz-vsub">${esc(d.subtitle)}</p>
        <div class="kz-vans">${esc(d.answer)}</div>
        ${hard ? `<div class="kz-hard"><div class="t">${esc(hard.kicker)}</div><div style="margin-top:7px">${esc(hard.text)}</div></div>` : ""}
      </div>
      <div class="kz-metrics">
        <div class="kz-metric"><div class="l">Saha durumu</div><div class="v">${esc(d.situation)}</div></div>
        <div class="kz-metric"><div class="l">Karar</div><div class="v">${esc(d.decision)}</div></div>
        <div class="kz-metric"><div class="l">Analiz süresi</div><div class="v big">${(d.total_s * p).toFixed(1)} sn</div></div>
        <div class="kz-metric"><div class="l">İşaretlenen olay</div><div class="v big">${Math.round(events.length * p)}</div></div>
      </div>
      <div class="kz-split">
        <section class="kz-card">
          <h2 class="kz-h2">Kayıt ve kanıt kareleri</h2>
          <div class="kz-player">${d.video_url ? `<video src="${esc(d.video_url)}" controls></video>` : `<div class="kz-help" style="padding:40px;text-align:center">${esc(d.video_name || "kayıt")}</div>`}</div>
          <div class="kz-frames">${frameHtml}</div>
        </section>
        <section>
          <div class="kz-card">
            <h2 class="kz-h2">Olay zaman çizelgesi</h2>
            ${events.length ? `<div class="kz-tl">${events.map((ev) => `<div><time>${esc(ev.time)}</time><span>${esc(ev.event)}</span></div>`).join("")}</div>` : '<div class="kz-help" style="margin-top:14px">Ayrı bir olay satırı işaretlenmedi; rutin akış.</div>'}
          </div>
          <div class="kz-card" style="margin-top:22px">
            <h2 class="kz-h2">Özet</h2>
            <p style="margin:12px 0 0;font-size:15.5px;line-height:1.65;color:var(--kz-code-text)">${esc(d.summary)}</p>
          </div>
          <div class="kz-card" style="margin-top:22px">
            <h2 class="kz-h2">Saha aksiyonları</h2>
            <div class="kz-acts">${actions.map((text, i) => `<div><i>${String(i + 1).padStart(2, "0")}</i><span>${esc(text)}</span></div>`).join("")}</div>
          </div>
          <div class="kz-acc" data-acc="json">
            <button type="button"><span class="kz-caret ${S.jsonOpen ? "open" : ""}">▸</span> Jüri çıktısı (şartname JSON)</button>
            <div class="kz-acc-body ${S.jsonOpen ? "" : "is-off"}">
              <div class="kz-help" style="padding-bottom:12px">${esc(d.spec_footnote || "")}</div>
              <pre>${esc(JSON.stringify(d.spec || { summary: d.summary, events, risk: d.tone, actions }, null, 2))}</pre>
            </div>
          </div>
          <div class="kz-acc" data-acc="timings">
            <button type="button"><span class="kz-caret ${S.timingsOpen ? "open" : ""}">▸</span> Aşama süreleri ve model çağrıları</button>
            <div class="kz-acc-body ${S.timingsOpen ? "" : "is-off"}">
              ${Object.entries(d.timings || {}).map(([key, value]) => `<div class="kz-row"><span class="kz-help">${esc(key)}</span><span>${Number(value).toFixed(2)} sn</span></div>`).join("")}
            </div>
          </div>
          <div class="kz-acc" data-acc="evidence">
            <button type="button"><span class="kz-caret ${S.evidenceOpen ? "open" : ""}">▸</span> Sensör kanıtı (hareket / yakınlık / yangın)</button>
            <div class="kz-acc-body ${S.evidenceOpen ? "" : "is-off"}"><pre>${esc(JSON.stringify(d.evidence || {}, null, 2))}</pre></div>
          </div>
          <div class="kz-acc gold" data-acc="law">
            <button type="button"><span class="kz-caret ${S.lawOpen ? "open" : ""}">▸</span> ${esc((law && law.kicker) || d.law_note || "Mevzuat")}</button>
            <div class="kz-acc-body ${S.lawOpen ? "" : "is-off"}">
              ${((law && law.articles) || []).map((item) => `<div style="padding:10px 0 0;border-top:1px solid rgba(var(--kz-line-rgb),0.08)"><div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--kz-gold);margin-bottom:6px">${esc(item.title)}</div><p style="margin:0;font-size:13.5px;line-height:1.55">${esc(item.text)}</p></div>`).join("")}
            </div>
          </div>
          <div class="kz-dl">
            <a href="#" data-dl="spec">Şartname JSON indir</a>
            <a class="sec" href="#" data-dl="full">Tam sonuç JSON indir</a>
          </div>
        </section>
      </div>
    </div>
  </section>`;
}

function smoothScrollTo(el) {
  if (!el) return;
  const top = el.getBoundingClientRect().top + window.pageYOffset - 28;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

function hideOverlay() {
  return new Promise((resolve) => {
    const overlay = $("overlay");
    if (!overlay || overlay.classList.contains("is-off")) {
      resolve();
      return;
    }
    overlay.classList.add("leaving");
    setTimeout(() => {
      overlay.classList.add("is-off");
      overlay.classList.remove("leaving");
      overlay.innerHTML = "";
      resolve();
    }, 160);
  });
}

function liveEvents(snap, phase) {
  const spec = (snap && snap.spec) || {};
  const raw = spec.events || [];
  const stamp = String((snap && (snap.event_time || snap.trigger_time)) || "").trim();
  const summary = String(spec.summary || "").trim();
  const rows = [];
  const seen = new Set();
  const push = (time, event) => {
    const t = String(time || "").trim();
    const e = String(event || "").trim();
    if (!e) return;
    const key = t + "|" + e;
    if (seen.has(key)) return;
    seen.add(key);
    rows.push({ time: t || "00:00", event: e });
  };
  if (!raw.length && stamp && summary) push(stamp, summary);
  raw.forEach((ev) => push(ev && ev.time, ev && ev.event));
  if (!rows.length && stamp && (phase === "candidate" || phase === "analyzing")) {
    push(stamp, "Hareket tetiklendi — model bu pencereyi okuyor");
  }
  if (stamp && rows.length && rows[0].time !== stamp) {
    const idx = rows.findIndex((row) => row.time === stamp);
    if (idx > 0) {
      const hit = rows.splice(idx, 1)[0];
      rows.unshift(hit);
    }
  }
  return rows;
}

function renderLive(snap) {
  const running = !!(snap && snap.running);
  const phase = (snap && snap.phase) || "idle";
  const banner = (snap && snap.banner) || {};
  const spec = (snap && snap.spec) || {};
  const actions = spec.actions || [];
  const src = (snap && snap.source_info) || { label: "EVREN", tone: "ok", detail: "vlm · gerçek zamanlı" };
  const tone = banner.tone || "ok";
  $("live-cap").textContent = running
    ? "Akış açık — soldaki görüntü canlı; model kısa klibi arkada okur."
    : "Akış kapalı. Video seçin veya webcam açın, sonra başlatın.";
  const decided = phase === "decided" || !!(snap && snap.has_brief) || !!(spec.summary || actions.length || (spec.events || []).length);
  const events = liveEvents(snap, phase);
  let eventHtml = '<div class="kz-help">Henüz tetik yok. Hareket olunca saniye burada durur.</div>';
  if (events.length) {
    eventHtml = `<div class="kz-tl">${events.map((ev) => `<div><time>${esc(ev.time)}</time><span>${esc(ev.event)}</span></div>`).join("")}</div>`;
  }
  let summaryHtml = "Tetik yok. Olay özeti burada durur.";
  if (decided && spec.summary) summaryHtml = spec.summary;
  else if (phase === "candidate" || phase === "analyzing") summaryHtml = "Özet model dönünce yazılır. Akış durmadı.";
  const metrics = decided
    ? [
        ["Algılanan an", (snap && (snap.event_time || snap.trigger_time)) || "—"],
        ["Saha durumu", banner.situation || "—"],
        ["Karar", banner.decision || "—"],
        ["Model", snap && snap.latency_s ? Number(snap.latency_s).toFixed(0) + " sn" : "—"]
      ]
    : [
        ["Durum", banner.kicker || phase],
        ["Hareket", String(Math.round(snap && snap.motion_score ? snap.motion_score : 0))],
        ["Tetik", String((snap && snap.triggers) || 0)],
        ["Algılanan an", (snap && snap.trigger_time) || "—"]
      ];
  const law = snap && snap.law;
  let actionHtml = '<div class="kz-help">Tetik yok. Aksiyon listesi burada durur.</div>';
  if (decided && actions.length) {
    actionHtml = `<div class="kz-acts">${actions.map((text, i) => `<div><i>${String(i + 1).padStart(2, "0")}</i><span>${esc(text)}</span></div>`).join("")}</div>`;
  } else if (phase === "candidate" || phase === "analyzing") {
    actionHtml = '<div class="kz-help">Aksiyon önerisi karar ile birlikte gelir.</div>';
  }
  $("live-body").innerHTML = `<div class="kz-live-grid">
    <section class="kz-card">
      <div class="kz-kicker-sm">Kayıt</div>
      <div class="kz-feed">
        ${running ? '<div class="kz-live-tag"><i></i>CANLI</div>' : ""}
        <img id="live-img" alt="">
        <div class="kz-help" id="live-empty">${running ? "Kare bekleniyor…" : "Kayıt henüz kare göndermedi. İzlemeyi başlatın."}</div>
      </div>
    </section>
    <section class="kz-verdict ${esc(tone)}" style="margin:0;animation:none">
      <div class="kz-source ${esc(src.tone || "ok")}">Kaynak · ${esc(src.label || "EVREN")}<span> ${esc(src.detail || "")}</span></div>
      <div class="kz-vkick">${esc(banner.kicker || "Operatör konsolu")}</div>
      <div class="kz-h2" style="margin:10px 0 6px;font-size:26px">${esc(banner.title || "Kamera bekleniyor")}</div>
      <p class="kz-vsub" style="font-size:14.5px">${esc(banner.subtitle || "")}</p>
      ${snap && snap.error ? `<p class="kz-err" style="margin-top:12px">${esc(snap.error)}</p>` : ""}
      <div class="kz-metrics" style="margin-top:16px;grid-template-columns:repeat(2,minmax(0,1fr))">
        ${metrics.map((cell) => `<div class="kz-metric"><div class="l">${esc(cell[0])}</div><div class="v">${esc(cell[1])}</div></div>`).join("")}
      </div>
    </section>
  </div>
  <div class="kz-brief">
    <div class="kz-split">
      <div>
        <div class="kz-kicker-sm" style="color:var(--kz-gold);margin-bottom:12px">Olay zamanı${snap && snap.event_time ? " · " + esc(snap.event_time) : ""}</div>
        ${eventHtml}
      </div>
      <div>
        <div class="kz-kicker-sm" style="color:var(--kz-gold);margin-bottom:12px">Olay özeti</div>
        <div style="font-size:14px;line-height:1.55;color:var(--kz-muted)">${esc(summaryHtml)}</div>
      </div>
    </div>
    <div style="margin-top:22px">
      <div class="kz-kicker-sm" style="color:var(--kz-gold);margin-bottom:12px">Saha aksiyonları</div>
      ${actionHtml}
    </div>
    ${law && law.articles ? `<div class="kz-acc gold" style="margin-top:18px" data-acc="live-law"><button type="button"><span class="kz-caret">▸</span> ${esc(law.kicker)}</button><div class="kz-acc-body is-off">${law.articles.map((item) => `<div style="padding:10px 0 0"><div class="kz-kicker-sm">${esc(item.title)}</div><p>${esc(item.text)}</p></div>`).join("")}</div></div>` : ""}
  </div>`;
}

async function playFlow(resultPromise) {
  clearTimers();
  S.phase = "running";
  S.waitingResult = true;
  S.holdOverlay = false;
  S.doneSteps = 0;
  S.countP = 1;
  $("result").innerHTML = "";
  paintChrome();
  renderOverlay();
  let data = null;
  const fetchP = Promise.resolve(resultPromise).then((row) => { data = row; return row; });
  for (let i = 0; i < 5; i += 1) {
    await sleep(stepMs(i));
    S.doneSteps = i + 1;
    renderOverlay();
  }
  if (!data) {
    S.holdOverlay = true;
    renderOverlay();
  }
  try {
    data = await fetchP;
  } catch (err) {
    S.holdOverlay = false;
    S.waitingResult = false;
    S.phase = "idle";
    await hideOverlay();
    paintChrome();
    showErr(String(err.message || err));
    return;
  }
  if (!data || data.ok === false) {
    S.holdOverlay = false;
    S.waitingResult = false;
    S.phase = "idle";
    await hideOverlay();
    paintChrome();
    showErr((data && data.error) || "Analiz tamamlanamadı");
    return;
  }
  S.result = data;
  S.holdOverlay = false;
  S.waitingResult = false;
  S.phase = "done";
  paintChrome();
  renderResult();
  smoothScrollTo($("result"));
  await hideOverlay();
}

function showLiveErr(text) {
  const box = $("live-err");
  if (!box) return;
  box.textContent = text;
  box.classList.toggle("is-off", !text);
}

function liveFingerprint(snap) {
  if (!snap) return "";
  return [
    snap.running,
    snap.phase,
    snap.trigger_time,
    snap.event_time,
    snap.triggers,
    snap.latency_s,
    snap.error,
    snap.banner && snap.banner.title,
    JSON.stringify(snap.spec || {})
  ].join("|");
}

async function refreshLivePreview() {
  const img = $("live-img");
  if (!img) return;
  try {
    const res = await fetch("/api/live/preview?t=" + Date.now(), { cache: "no-store" });
    if (!res.ok) return;
    const blob = await res.blob();
    if (blob.size < 32) return;
    if (blob.type && blob.type.indexOf("json") >= 0) return;
    const url = URL.createObjectURL(blob);
    img.src = url;
    img.classList.add("is-on");
    const empty = $("live-empty");
    if (empty) empty.classList.add("is-off");
    if (livePreviewUrl) URL.revokeObjectURL(livePreviewUrl);
    livePreviewUrl = url;
  } catch (_err) {
    /* kare henüz yok */
  }
}

async function pollLiveOnce() {
  if (S.view !== "live") return;
  try {
    const snap = await (await fetch("/api/live/status", { cache: "no-store" })).json();
    S.liveSnap = snap;
    const fp = liveFingerprint(snap);
    if (fp !== liveFp) {
      liveFp = fp;
      renderLive(snap);
    }
    if (snap.running) await refreshLivePreview();
  } catch (_err) {
    /* çevrimdışı */
  }
}

function startLivePoll() {
  stopLivePoll();
  pollLiveOnce();
  livePoll = window.setInterval(pollLiveOnce, 280);
}

function stopLivePoll() {
  if (livePoll) window.clearInterval(livePoll);
  livePoll = 0;
}

async function startLive() {
  showLiveErr("");
  const body = new FormData();
  body.append("kind", S.liveKind);
  body.append("lock", S.lock ? "1" : "0");
  body.append("provider", S.provider);
  body.append("motion", $("live-motion") ? $("live-motion").value : "12");
  body.append("cooldown", $("live-cooldown") ? $("live-cooldown").value : "8");
  if (S.liveKind === "webcam") {
    body.append("cam", $("live-cam") ? $("live-cam").value : "0");
  } else {
    if (!S.liveFile) {
      showLiveErr("Önce bir video seçin.");
      return;
    }
    body.append("video", S.liveFile, S.liveFile.name);
  }
  try {
    const res = await fetch("/api/live/start", { method: "POST", body });
    const data = await res.json();
    if (!data.ok) {
      showLiveErr(data.error || "İzleme başlatılamadı.");
      return;
    }
    liveFp = "";
    S.liveSnap = data;
    renderLive(data);
    startLivePoll();
  } catch (_err) {
    showLiveErr("Sunucuya bağlanılamadı. jury_server.py çalışıyor olsun.");
  }
}

async function stopLive() {
  showLiveErr("");
  try {
    const data = await (await fetch("/api/live/stop", { method: "POST" })).json();
    liveFp = "";
    S.liveSnap = data;
    renderLive(data);
  } catch (_err) {
    showLiveErr("Durdurma isteği gönderilemedi.");
  }
}

function showErr(text) {
  const box = $("err");
  box.textContent = text;
  box.classList.toggle("is-off", !text);
}

function onLiveFile(file) {
  S.liveFile = file;
  if (liveFileBlob) URL.revokeObjectURL(liveFileBlob);
  liveFileBlob = URL.createObjectURL(file);
  const preview = $("live-preview");
  if (preview) preview.src = liveFileBlob;
  const name = $("live-file-name");
  if (name) name.textContent = file.name;
  $("live-drop-empty").classList.add("is-off");
  $("live-drop-loaded").classList.remove("is-off");
  $("live-drop").classList.add("has-file");
  showLiveErr("");
}

function onFile(file) {
  S.file = file;
  if (fileBlob) URL.revokeObjectURL(fileBlob);
  fileBlob = URL.createObjectURL(file);
  $("preview").src = fileBlob;
  $("drop-name").textContent = file.name;
  $("drop-empty").classList.add("is-off");
  $("drop-loaded").classList.remove("is-off");
  $("drop").classList.add("has-file");
  showErr("");
}

async function runReal() {
  if (!S.file) {
    showErr("Önce bir video yükleyin veya klasörden seçin.");
    return;
  }
  S.backup = false;
  S.skipSecond = S.fast;
  const body = new FormData();
  body.append("video", S.file, S.file.name);
  body.append("prompt", $("prompt").value);
  body.append("lock", S.lock ? "1" : "0");
  body.append("provider", S.provider);
  body.append("fast", S.fast ? "1" : "0");
  body.append("rag", S.rag ? "1" : "0");
  body.append("frames", String(S.frames));
  const pending = fetch("/api/analyze", { method: "POST", body }).then(async (res) => {
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (_err) {
      return {
        ok: false,
        error: "Sunucu analiz yanıtını gönderemedi. http://127.0.0.1:8503 üzerinden açın; jury_server.py çalışıyor olsun.",
      };
    }
    if (!res.ok && !data.error) data.error = "HTTP " + res.status;
    return data;
  }).catch(() => ({
    ok: false,
    error: "Sunucuya bağlanılamadı. Tarayıcıda http://127.0.0.1:8503 açık olsun ve `py app/jury_server.py` çalışsın.",
  }));
  await playFlow(pending);
}

function downloadJson(kind) {
  if (!S.result) return;
  const payload = kind === "spec"
    ? (S.result.spec || {})
    : S.result;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (S.result.video_name || "kaizen").replace(/\.[^.]+$/, "") + "_" + kind + ".json";
  a.click();
}

async function boot() {
  try {
    const data = await (await fetch("/api/boot")).json();
    if (data.prompt) $("prompt").value = data.prompt;
    if (typeof data.frames === "number") {
      S.frames = data.frames;
      $("frames").value = String(data.frames);
    }
    if (typeof data.fast === "boolean") S.fast = data.fast;
    $("describe").textContent = data.describe || $("describe").textContent;
    const sel = $("runs");
    (data.runs || []).forEach((run) => {
      const opt = document.createElement("option");
      opt.value = run.name;
      opt.textContent = run.name;
      sel.appendChild(opt);
    });
  } catch (_err) {
    /* çevrimdışı maket */
  }
  paintChrome();
  setTheme(S.theme);
}

document.addEventListener("click", (event) => {
  const acc = event.target.closest("[data-acc]");
  if (acc && event.target.closest("button")) {
    const key = acc.dataset.acc;
    if (key === "live-law") {
      const body = acc.querySelector(".kz-acc-body");
      const caret = acc.querySelector(".kz-caret");
      if (body) body.classList.toggle("is-off");
      if (caret) caret.classList.toggle("open");
      return;
    }
    if (key === "json") S.jsonOpen = !S.jsonOpen;
    if (key === "timings") S.timingsOpen = !S.timingsOpen;
    if (key === "evidence") S.evidenceOpen = !S.evidenceOpen;
    if (key === "law") S.lawOpen = !S.lawOpen;
    renderResult();
    return;
  }
  const dl = event.target.closest("[data-dl]");
  if (dl) {
    event.preventDefault();
    downloadJson(dl.dataset.dl);
    return;
  }
  const act = event.target.closest("[data-act]");
  if (!act) return;
  const name = act.dataset.act;
  if (name === "theme") setTheme(act.dataset.val);
  if (name === "sidebar") {
    S.sidebar = !S.sidebar;
    paintChrome();
  }
  if (name === "view") {
    S.view = act.dataset.val;
    paintChrome();
    if (S.view === "live") {
      renderLive(S.liveSnap);
      startLivePoll();
    } else {
      stopLivePoll();
    }
  }
  if (name === "lock") {
    S.lock = !S.lock;
    paintChrome();
  }
  if (name === "fast") {
    S.fast = !S.fast;
    paintChrome();
  }
  if (name === "rag") {
    S.rag = !S.rag;
    paintChrome();
  }
  if (name === "provider") {
    S.provider = act.dataset.val;
    paintChrome();
  }
  if (name === "analyze") runReal();
  if (name === "backup") {
    const runName = $("runs").value;
    if (!runName) return;
    S.backup = true;
    playFlow(fetch("/api/run/" + encodeURIComponent(runName)).then((res) => res.json()));
  }
  if (name === "live-kind") {
    S.liveKind = act.dataset.val;
    document.querySelectorAll("[data-act=live-kind]").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.val === S.liveKind);
    });
    $("live-file-box").classList.toggle("is-off", S.liveKind !== "dosya");
    $("live-cam-box").classList.toggle("is-off", S.liveKind !== "webcam");
  }
  if (name === "live-start") startLive();
  if (name === "live-stop") stopLive();
});

$("frames").addEventListener("input", (event) => {
  S.frames = Number(event.target.value);
  paintChrome();
});
$("file").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) onFile(file);
});
$("drop").addEventListener("click", (event) => {
  if (event.target.closest("video") || event.target.id === "file") return;
  $("file").click();
});
["dragenter", "dragover"].forEach((type) => {
  $("drop").addEventListener(type, (event) => { event.preventDefault(); });
});
$("drop").addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) onFile(file);
});

const liveDrop = $("live-drop");
const liveFileInput = $("live-file");
if (liveDrop && liveFileInput) {
  liveFileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) onLiveFile(file);
  });
  liveDrop.addEventListener("click", (event) => {
    if (event.target.closest("video") || event.target.id === "live-file") return;
    liveFileInput.click();
  });
  ["dragenter", "dragover"].forEach((type) => {
    liveDrop.addEventListener(type, (event) => { event.preventDefault(); });
  });
  liveDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) onLiveFile(file);
  });
}
const liveMotion = $("live-motion");
if (liveMotion) {
  liveMotion.addEventListener("input", (event) => {
    $("live-motion-val").textContent = String(event.target.value);
  });
}
const liveCooldown = $("live-cooldown");
if (liveCooldown) {
  liveCooldown.addEventListener("input", (event) => {
    $("live-cd-val").textContent = String(event.target.value);
  });
}

boot();
