const $ = (selector) => document.querySelector(selector);
const state = { products: [], orders: [], messages: [], lastOpen: 0, recorder: null, voiceStream: null, audioUrl: null, audioContext: null, meterFrame: null, voiceText: "", voiceSession: 0, recording: false, attachment: null };
const suggestions = ["Kain dia panas tak?", "Jahitan kemas ke?", "Warna okay tak?", "Where is my order?"];
const palette = { navy: ["#143a72", "#315ca1"], sage: ["#6e1733", "#9c3b58"], sand: ["#a32135", "#d44752"] };

function garmentSvg(style) {
  const [base, trim] = palette[style] || palette.navy;
  return `<svg viewBox="0 0 300 390" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="cloth-${style}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${trim}"/><stop offset="1" stop-color="${base}"/></linearGradient></defs>
    <path d="M105 25 80 41 33 62 8 157 54 172 79 118 80 229 55 350 Q150 377 245 350 L220 229 221 118 246 172 292 157 267 62 220 41 195 25 176 46 150 55 124 46Z" fill="url(#cloth-${style})" stroke="${base}" stroke-width="2"/>
    <path d="M105 25Q123 57 150 61Q177 57 195 25L178 18Q168 41 150 43Q132 41 122 18Z" fill="${base}" stroke="${trim}" stroke-width="2"/>
    <path d="M150 58V240" stroke="${trim}" stroke-width="3"/><circle cx="150" cy="90" r="3" fill="#e7ddc7"/><circle cx="150" cy="125" r="3" fill="#e7ddc7"/><circle cx="150" cy="160" r="3" fill="#e7ddc7"/><circle cx="150" cy="195" r="3" fill="#e7ddc7"/>
    <path d="M78 225Q150 239 222 225M61 338Q150 357 239 338M28 151 56 158M244 158 272 151" fill="none" stroke="${trim}" stroke-width="3" opacity=".6"/>
  </svg>`;
}

function productImage(product, className = "garment") {
  const image = `/assets/${product.id}.jpg`;
  return `<div class="${className}"><img src="${image}" alt="${product.name}" onload="this.nextElementSibling.hidden=true" onerror="this.style.display='none'"/><span class="garment-fallback">${garmentSvg(product.style)}</span></div>`;
}

function toast(text) {
  const node = $("#toast"); node.textContent = text; node.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 2600);
}

function showPage(page) {
  document.querySelectorAll(".page").forEach(node => node.classList.toggle("active", node.id === `${page}-page`));
  document.querySelectorAll(".site-header nav button").forEach(node => node.classList.toggle("nav-active", node.dataset.page === page));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderProducts() {
  $("#hero-garment").innerHTML = `<img src="/assets/hero.jpg?v=2" alt="Woman wearing Rojak Wardrobe clothing" onload="this.nextElementSibling.hidden=true" onerror="this.style.display='none'"><span class="garment-fallback">${garmentSvg("navy")}</span>`;
  const root = $("#product-grid"); root.replaceChildren();
  state.products.forEach(product => {
    const card = document.createElement("article"); card.className = "product-card";
    card.innerHTML = `<div class="product-image ${product.style}"><span class="tag">${product.tag}</span>${productImage(product)}</div><div class="product-info"><h3></h3><div class="product-meta"></div><div class="product-bottom"><strong></strong><button type="button">Ask about collection</button></div></div>`;
    card.querySelector("h3").textContent = product.name;
    card.querySelector(".product-meta").textContent = product.colour;
    card.querySelector("strong").textContent = `RM ${product.price.toFixed(2)}`;
    card.querySelector("button").addEventListener("click", () => openChat({ kind: "product", id: product.id, name: product.name, variant: product.colour, style: product.style }));
    root.append(card);
  });
}

function copyCode(code) {
  navigator.clipboard.writeText(code).then(() => toast("Code copied to clipboard")).catch(() => toast(`Select and copy: ${code}`));
}

function renderOrders() {
  const root = $("#orders-list"); root.replaceChildren();
  state.orders.forEach(order => {
    const card = document.createElement("article"); card.className = "order-card";
    card.innerHTML = `<div><span class="status"></span><h2></h2><p class="order-sub"></p><strong></strong><div class="order-codes"><span>ORDER ID<code></code></span><span>PARCEL CODE<code></code></span></div></div><div class="order-actions"><button class="view">View details</button><button class="copy">Copy order ID</button></div>`;
    card.querySelector(".status").textContent = order.status;
    card.querySelector("h2").textContent = order.product;
    card.querySelector(".order-sub").textContent = `${order.variant} · Ordered ${order.date}`;
    card.querySelector("strong").textContent = `RM ${order.total.toFixed(2)}`;
    const codes = card.querySelectorAll("code"); codes[0].textContent = order.id; codes[1].textContent = order.tracking;
    card.querySelector(".view").addEventListener("click", () => renderOrderDetail(order));
    card.querySelector(".copy").addEventListener("click", () => copyCode(order.id));
    root.append(card);
  });
}

function renderOrderDetail(order) {
  $("#order-title").textContent = order.id;
  const root = $("#order-detail"); root.replaceChildren();
  const card = document.createElement("div"); card.className = "order-detail-card";
  card.innerHTML = `<div><span class="status"></span><h2></h2><p class="order-sub"></p><div class="detail-row">Order ID<strong class="order-id"></strong></div><div class="detail-row">Parcel code<strong class="tracking"></strong></div><div class="detail-row">Expected / completed<strong class="eta"></strong></div><div class="detail-row">Total<strong class="total"></strong></div><div class="order-actions" style="margin-top:28px"><button class="copy-order">Copy order ID</button><button class="copy-parcel">Copy parcel code</button><button class="support">Ask customer support</button></div></div><div><h2>Order journey</h2><div class="timeline"></div></div>`;
  card.querySelector(".status").textContent = order.status;
  card.querySelector("h2").textContent = order.product;
  card.querySelector(".order-sub").textContent = order.variant;
  card.querySelector(".order-id").textContent = order.id;
  card.querySelector(".tracking").textContent = order.tracking;
  card.querySelector(".eta").textContent = order.eta;
  card.querySelector(".total").textContent = `RM ${order.total.toFixed(2)}`;
  order.steps.forEach(step => { const line = document.createElement("div"); line.textContent = step; card.querySelector(".timeline").append(line); });
  card.querySelector(".copy-order").addEventListener("click", () => copyCode(order.id));
  card.querySelector(".copy-parcel").addEventListener("click", () => copyCode(order.tracking));
  card.querySelector(".support").addEventListener("click", () => openChat({ kind: "order", id: order.id, name: order.product, variant: order.variant, style: state.products.find(p => p.name === order.product)?.style || "navy" }));
  root.append(card); showPage("order");
}

function saveChat() {
  try { localStorage.setItem("rojak-chat-v1", JSON.stringify({ messages: state.messages.slice(-40), lastOpen: state.lastOpen })); } catch (_) {}
}

function renderAttachment() {
  const box = $("#chat-attachment"); box.replaceChildren();
  const item = state.attachment;
  box.hidden = !item;
  if (!item) return;
  const card = document.createElement("div"); card.className = "attached-card";
  const art = document.createElement("div"); art.className = `attached-art ${item.style || "navy"}`;
  if (item.kind === "product") art.innerHTML = productImage(item, "attached-image");
  else art.textContent = "▣";
  const info = document.createElement("div"); info.className = "attached-info";
  const label = document.createElement("span"); label.textContent = item.kind === "order" ? "ORDER ENQUIRY" : "PRODUCT ENQUIRY";
  const title = document.createElement("strong"); title.textContent = item.name;
  const detail = document.createElement("small"); detail.textContent = item.kind === "order" ? `${item.id} · ${item.variant}` : item.variant;
  info.append(label, title, detail);
  const close = document.createElement("button"); close.className = "attachment-remove"; close.type = "button"; close.textContent = "×"; close.setAttribute("aria-label", "Remove attached item");
  close.addEventListener("click", () => { state.attachment = null; renderAttachment(); $("#chat-input").focus(); });
  card.append(art, info, close); box.append(card);
}

function renderMessages() {
  const root = $("#chat-messages"); root.replaceChildren();
  const chip = document.createElement("div"); chip.className = "date-chip"; chip.textContent = "Today · Rojak Wardrobe"; root.append(chip);
  const lastGreeting = state.messages.findLastIndex(message => message.quickReplies);
  const promptVisible = lastGreeting >= 0 && !state.messages.slice(lastGreeting + 1).some(message => message.role === "customer");
  state.messages.forEach((message, index) => {
    const row = document.createElement("div"); row.className = `msg ${message.role}`;
    const icon = document.createElement("span"); icon.className = "msg-icon"; icon.textContent = message.role === "customer" ? "👤" : "✳";
    const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = message.text;
    if (message.attachment) {
      const mini = document.createElement("div"); mini.className = "sent-attachment";
      const thumb = document.createElement("div"); thumb.className = `sent-thumb ${message.attachment.style || "navy"}`;
      if (message.attachment.kind === "product") thumb.innerHTML = productImage(message.attachment, "attached-image");
      else thumb.textContent = "▣";
      const info = document.createElement("div");
      const name = document.createElement("strong"); name.textContent = message.attachment.name;
      const code = document.createElement("small"); code.textContent = message.attachment.kind === "order" ? `Order ${message.attachment.id}` : message.attachment.variant;
      info.append(name, code); mini.append(thumb, info);
      bubble.prepend(mini);
    }
    if (message.role === "customer") row.append(bubble, icon); else row.append(icon, bubble);
    root.append(row);
    if (index === lastGreeting && promptVisible) {
      const choices = document.createElement("div"); choices.className = "quick-replies";
      suggestions.forEach(text => { const button = document.createElement("button"); button.textContent = text; button.addEventListener("click", () => sendMessage(text)); choices.append(button); });
      root.append(choices);
    }
    if (message.evidence?.length) {
      const details = document.createElement("details"); details.className = "evidence";
      const summary = document.createElement("summary"); summary.textContent = `View ${message.evidence.length} customer reviews`; details.append(summary);
      message.evidence.forEach(hit => { const quote = document.createElement("blockquote"); quote.textContent = `${hit.rating ?? "–"}★  ${hit.review}`; details.append(quote); });
      root.append(details);
    }
    if (message.role === "assistant" && index > 0 && !message.quickReplies) {
      const feedback = document.createElement("div"); feedback.className = "feedback-wrap";
      const trigger = document.createElement("button"); trigger.className = "feedback-trigger"; trigger.textContent = "Correct this answer";
      trigger.addEventListener("click", () => {
        if (feedback.querySelector("form")) { feedback.querySelector("form").remove(); return; }
        const form = document.createElement("form"); form.className = "feedback-form";
        const field = document.createElement("textarea"); field.placeholder = "What should the assistant have answered?"; field.required = true; field.maxLength = 2000;
        const submit = document.createElement("button"); submit.type = "submit"; submit.textContent = "Submit correction";
        form.append(field, submit); feedback.append(form); field.focus();
        form.addEventListener("submit", async event => {
          event.preventDefault();
          const previous = state.messages.slice(0, index).reverse().find(item => item.role === "customer");
          try {
            const result = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: previous?.text || "", correction: field.value }) });
            if (!result.ok) throw new Error();
            form.remove(); toast("Correction saved for review");
          } catch (_) { toast("Could not save correction"); }
        });
      });
      feedback.append(trigger);
      if (message.suggestions?.length) {
        const prompt = document.createElement("div"); prompt.className = "suggestion-prompt";
        prompt.textContent = "Is this what you meant instead?";
        const options = document.createElement("div"); options.className = "quick-replies";
        message.suggestions.forEach(option => {
          const choice = document.createElement("button"); choice.type = "button";
          choice.textContent = option.label;
          choice.addEventListener("click", () => {
            trigger.click();
            const field = feedback.querySelector("textarea");
            if (field) { field.value = "I meant " + option.label + "."; field.focus(); }
          });
          options.append(choice);
        });
        feedback.append(prompt, options);
      }
      root.append(feedback);
    }
  });
  root.scrollTop = root.scrollHeight;
}

function appendMessage(role, text, extra = {}) {
  state.messages.push({ role, text, ...extra }); saveChat(); renderMessages();
}

function openChat(attachment = null) {
  const now = Date.now();
  if (!state.messages.length || now - state.lastOpen > 30 * 60 * 1000) {
    appendMessage("assistant", "Hi Aina! 👋 Welcome to Rojak Wardrobe. What can I help you with today? Ask in English, BM or Rojak.", { quickReplies: true });
  }
  state.lastOpen = now; saveChat();
  // Repaint restored local history whenever the chat opens; do not wait for
  // the next message append to trigger the first render.
  renderMessages();
  $("#chat-panel").classList.add("open"); $("#chat-panel").setAttribute("aria-hidden", "false");
  $("#floating-chat").hidden = true;
  if (attachment) {
    state.attachment = attachment;
    renderAttachment();
    appendMessage("assistant", attachment.kind === "order" ? `What would you like to know about your ${attachment.name} order?` : `What would you like to know about ${attachment.name}? You can ask about fabric, stitching or colour.`);
  }
  $("#chat-input").focus();
}

function closeChat() {
  discardVoice();
  $("#chat-panel").classList.remove("open"); $("#chat-panel").setAttribute("aria-hidden", "true");
  $("#floating-chat").hidden = false;
}

async function sendMessage(preset = null) {
  const input = $("#chat-input"); const text = (preset || input.value).trim();
  if (!text) return;
  const attached = state.attachment;
  input.value = ""; appendMessage("customer", text, attached ? { attachment: attached } : {});
  state.attachment = null; renderAttachment();
  const send = $("#send-button"); send.disabled = true;
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text, context: attached ? { kind: attached.kind, id: attached.id } : null }) });
    if (!response.ok) throw new Error("Chat service unavailable");
    const data = await response.json(); appendMessage("assistant", data.reply, { evidence: data.evidence || [], suggestions: data.suggestions || [], responseTimeMs: data.response_time_ms });
  } catch (_) { appendMessage("assistant", "Sorry, I couldn't connect to the shop assistant. Please try again."); }
  finally { send.disabled = false; input.focus(); }
}

function setVoiceStatus(text) { $("#voice-status").textContent = text; }

function startVoiceMeter(stream) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return;
  try {
    const context = new AudioContext(); state.audioContext = context;
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser(); analyser.fftSize = 256;
    source.connect(analyser);
    const levels = new Uint8Array(analyser.frequencyBinCount);
    const bars = [...document.querySelectorAll(".voice-wave i")];
    bars.forEach(bar => { bar.style.animation = "none"; });
    const update = () => {
      if (!state.recording) return;
      analyser.getByteFrequencyData(levels);
      const volume = levels.reduce((sum, value) => sum + value, 0) / levels.length;
      bars.forEach((bar, index) => { bar.style.height = `${Math.min(20, 4 + volume * (.22 + index * .05))}px`; });
      state.meterFrame = requestAnimationFrame(update);
    };
    update();
  } catch (_) { /* The CSS listening animation remains available. */ }
}

function stopVoiceMeter() {
  if (state.meterFrame) cancelAnimationFrame(state.meterFrame);
  state.meterFrame = null;
  if (state.audioContext) state.audioContext.close().catch(() => {});
  state.audioContext = null;
  document.querySelectorAll(".voice-wave i").forEach(bar => { bar.style.animation = ""; bar.style.height = ""; });
}

function stopVoice() {
  if (!state.recording) return;
  state.recording = false;
  stopVoiceMeter();
  $("#voice-preview").classList.remove("listening");
  $("#voice-button").classList.remove("recording");
  $("#voice-button").setAttribute("aria-pressed", "false");
  $("#voice-button").setAttribute("aria-label", "Speak a question");
  setVoiceStatus("Processing recording…");
  if (state.recorder?.state === "recording") state.recorder.stop();
  state.voiceStream?.getTracks().forEach(track => track.stop());
  state.voiceStream = null;
}

function discardVoice() {
  state.voiceSession++;
  if (state.recording) stopVoice();
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = null; state.voiceText = "";
  $("#voice-playback").pause(); $("#voice-playback").removeAttribute("src"); $("#voice-playback").hidden = true;
  $("#voice-use-text").hidden = true; $("#voice-preview").hidden = true;
}

async function startVoice() {
  if (state.recording) { stopVoice(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { toast("Audio recording is unavailable in this browser"); return; }
  discardVoice();
  const session = state.voiceSession;
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (_) { toast("Microphone permission is needed to record your voice"); return; }
  if (session !== state.voiceSession) { stream.getTracks().forEach(track => track.stop()); return; }
  let recorder;
  try { recorder = new MediaRecorder(stream); }
  catch (_) { stream.getTracks().forEach(track => track.stop()); toast("Audio recording is unavailable in this browser"); return; }
  state.voiceStream = stream; state.recording = true;
  $("#voice-preview").hidden = false; $("#voice-preview").classList.add("listening");
  $("#voice-button").classList.add("recording"); $("#voice-button").setAttribute("aria-pressed", "true"); $("#voice-button").setAttribute("aria-label", "Stop voice recording");
  $("#voice-transcript").textContent = "Speak now. Your words will appear here.";
  setVoiceStatus("Listening… tap the mic to stop");
  startVoiceMeter(stream);
  const chunks = [];
  state.recorder = recorder;
  recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
  recorder.onstop = async () => {
    if (session !== state.voiceSession) return;
    if (!chunks.length) { setVoiceStatus("No audio was captured. Try again."); return; }
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    const clip = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    state.audioUrl = URL.createObjectURL(clip);
    const playback = $("#voice-playback"); playback.src = state.audioUrl; playback.hidden = false;
    setVoiceStatus("Sending recording to Whisper…");
    $("#voice-transcript").textContent = "Transcribing your English, BM or mixed-language speech…";
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Could not read the recording"));
        reader.readAsDataURL(clip);
      });
      const audio = String(dataUrl).split(",")[1] || "";
      const response = await fetch("/api/transcribe", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({audio})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Whisper transcription failed");
      if (session !== state.voiceSession) return;
      state.voiceText = result.text || "";
      $("#voice-transcript").textContent = state.voiceText || "No speech detected.";
      $("#voice-use-text").hidden = !state.voiceText;
      setVoiceStatus("Whisper ready · " + (result.language || "language unknown") + " · " + ((result.processing_time_ms || 0) / 1000).toFixed(1) + "s");
    } catch (error) {
      $("#voice-transcript").textContent = error.message;
      $("#voice-use-text").hidden = true;
      setVoiceStatus("Transcription failed. You can still play the recording or try again.");
    }
  };
  recorder.start();
}

async function init() {
  try {
    const response = await fetch("/api/catalog"); if (!response.ok) throw new Error();
    const data = await response.json(); state.products = data.products; state.orders = data.orders;
    renderProducts(); renderOrders();
  } catch (_) { toast("Could not load the shop. Please restart the Python server."); }
  try { const saved = JSON.parse(localStorage.getItem("rojak-chat-v1") || "null"); if (saved) { state.messages = saved.messages || []; state.lastOpen = saved.lastOpen || 0; } } catch (_) {}
  document.querySelectorAll("[data-page]").forEach(button => button.addEventListener("click", () => { closeChat(); showPage(button.dataset.page); }));
  [["#header-support", null], ["#floating-chat", null], ["#band-support", null], ["#story-support", null]].forEach(([selector]) => $(selector).addEventListener("click", () => openChat()));
  $("#hero-shop").addEventListener("click", () => $("#collection").scrollIntoView({ behavior: "smooth" }));
  $("#chat-back").addEventListener("click", closeChat); $("#chat-close").addEventListener("click", closeChat);
  $("#send-button").addEventListener("click", () => sendMessage());
  $("#chat-input").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); sendMessage(); } });
  $("#voice-button").addEventListener("click", startVoice);
  $("#voice-discard").addEventListener("click", discardVoice);
  $("#voice-use-text").addEventListener("click", () => {
    if (!state.voiceText) return;
    const input = $("#chat-input"); input.value = [input.value.trim(), state.voiceText].filter(Boolean).join(" ");
    input.focus(); toast("Transcript added. Check it before sending.");
  });
  document.addEventListener("keydown", event => { if (event.key === "Escape") closeChat(); });
}

init();
