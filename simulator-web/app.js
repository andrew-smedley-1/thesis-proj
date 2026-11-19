const byId = (id) => document.getElementById(id);
const world = byId("world");
const cases = byId("test-case");
const message = byId("message");
const messages = byId("messages");
const logs = byId("logs");
const status = byId("status");
const logFilter = byId("log-filter");
const logRows = { gazebo: [], agent: [] };
const cameras = {
  wrist: {
    image: byId("wrist-feed"), placeholder: byId("wrist-placeholder"),
    endpoint: "/api/frame", name: "wrist camera",
  },
  overview: {
    image: byId("overview-feed"), placeholder: byId("overview-placeholder"),
    endpoint: "/api/overview", name: "Gazebo overview",
  },
};
const errors = byId("error");
const dialog = byId("image-dialog");
const emptyChat = messages.firstElementChild.cloneNode(true);
const emptyLogs = logs.firstElementChild.cloneNode(true);
const stopWorld = byId("stop-world");
const worldOptions = new Map();
const phaseLabels = {
  select: "Select a world",
  loading: "Starting simulator…",
  stopping: "Stopping simulator…",
  ready: "Ready",
  running: "Agent working…",
  reply: "Waiting for your reply",
  failed: "World unavailable",
  offline: "Console disconnected",
};
const clock = new Intl.DateTimeFormat(undefined, { timeStyle: "medium" });
let state = { world: null, phase: "select" };
let cursor = 0;
let stateRequest = 0;
let busy = false;
let initialized = false;
let replay = false;

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${response.status}).`);
  }
  return response;
}

function error(text = "") {
  errors.textContent = text;
  errors.hidden = !text;
}

function renderState() {
  const phase = state.phase;
  const locked = ["loading", "running", "reply", "stopping"].includes(phase);
  const usable = !busy && ["ready", "reply"].includes(phase);
  world.disabled = !initialized || busy || locked || phase === "offline";
  byId("load-world").disabled = world.disabled || !world.value;
  byId("load-world").textContent = world.value && world.value === state.world ? "Reload world" : "Load world";
  stopWorld.disabled = busy || !state.world || ["offline", "stopping"].includes(phase);
  cases.disabled = !usable || phase === "reply";
  message.disabled = !usable;
  byId("send").disabled = !usable || !message.value.trim();
  message.placeholder = phase === "reply" ? "Type your reply" : "Type an instruction";
  status.dataset.phase = phase;
  status.textContent = replay ? `Replay · ${phaseLabels[phase]}` : phaseLabels[phase];
  byId("active-world").textContent = state.world ? worldOptions.get(state.world)?.label || "" : "";
}

function append(container, element, limit) {
  const follow = container.scrollHeight - container.scrollTop - container.clientHeight < 42;
  container.append(element);
  while (container.childElementCount > limit) container.firstElementChild.remove();
  if (follow) container.scrollTop = container.scrollHeight;
}

function logVisible(source) {
  return logFilter.value === "combined" || logFilter.value === source;
}

function addLog(event) {
  const source = event.kind === "simulator" ? "gazebo" : "agent";
  const row = document.createElement("div");
  row.className = `log-row ${event.kind}`;
  row.hidden = !logVisible(source);
  const time = document.createElement("time");
  time.textContent = clock.format(event.time * 1000);
  time.dateTime = new Date(event.time * 1000).toISOString();
  const text = document.createElement("span");
  text.textContent = event.text;
  row.append(time, text);

  const follow = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 42;
  const history = logRows[source];
  history.push(row);
  logs.append(row);
  if (history.length > 500) history.shift().remove();
  if (follow) logs.scrollTop = logs.scrollHeight;
}

function addEvent(event) {
  if (event.kind === "log" || event.kind === "simulator") {
    addLog(event);
    return;
  }
  messages.querySelector(".empty")?.remove();
  if (event.kind === "image") {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = event.text;
    const button = document.createElement("button");
    button.className = "image-button";
    button.setAttribute("aria-label", `Enlarge ${event.text.toLowerCase()}`);
    const image = document.createElement("img");
    image.src = event.url;
    image.alt = event.text;
    image.addEventListener("load", () => {
      if (messages.scrollHeight - figure.offsetHeight - messages.scrollTop - messages.clientHeight < 42) {
        messages.scrollTop = messages.scrollHeight;
      }
    });
    button.append(image);
    button.addEventListener("click", () => {
      const expanded = byId("expanded-image");
      expanded.src = event.url;
      expanded.alt = event.text;
      dialog.showModal();
    });
    figure.append(button, caption);
    append(messages, figure, 100);
    return;
  }
  const bubble = document.createElement("p");
  bubble.className = `bubble ${event.kind}`;
  const author = document.createElement("span");
  author.className = "author";
  author.textContent = event.kind === "user" ? "You" : "Agent";
  bubble.append(author, document.createTextNode(event.text));
  append(messages, bubble, 100);
}

const matchingCase = new Option("Use the selected world’s test instruction", "");
function showMatchingCase() {
  matchingCase.value = worldOptions.get(state.world)?.sample || "";
  matchingCase.hidden = !matchingCase.value || replay;
}

function resetSessionView() {
  cursor = 0;
  logRows.gazebo.length = 0;
  logRows.agent.length = 0;
  logs.replaceChildren(emptyLogs.cloneNode(true));
  messages.replaceChildren(emptyChat.cloneNode(true));
  message.value = "";
  cases.selectedIndex = 0;
  if (dialog.open) dialog.close();
}

async function updateState() {
  const requestNumber = ++stateRequest;
  const params = new URLSearchParams({ after: cursor, session: state.session || "" });
  const response = await request(`/api/state?${params}`);
  const next = await response.json();
  if (requestNumber !== stateRequest) return;
  const changedSession = next.session !== state.session;
  if (changedSession) resetSessionView();
  if (changedSession || next.world !== state.world || state.phase === "offline") {
    world.value = next.world || "";
  }
  const focusReply = state.phase !== "reply" && next.phase === "reply";
  state = next;
  showMatchingCase();
  for (const event of next.events) {
    addEvent(event);
    cursor = event.id;
  }
  renderState();
  if (focusReply && !busy && document.activeElement === document.body) message.focus();
}

async function pollState() {
  try {
    if (!busy) await updateState();
  } catch {
    state.phase = "offline";
    renderState();
  } finally {
    setTimeout(pollState, 550);
  }
}

function clearFrame(camera, text) {
  const { image, placeholder } = camera;
  const previousUrl = image.getAttribute("src");
  image.hidden = true;
  image.removeAttribute("src");
  if (previousUrl) URL.revokeObjectURL(previousUrl);
  placeholder.textContent = text;
  placeholder.hidden = false;
}

async function pollFrame(camera) {
  const { world: currentWorld, phase, session } = state;
  const { endpoint, name, image, placeholder } = camera;
  try {
    if (!currentWorld || ["loading", "stopping", "offline", "failed"].includes(phase)) {
      const text = {
        loading: `Waiting for the ${name}…`,
        stopping: "Stopping the simulator…",
        offline: "Console disconnected.",
        failed: `The ${name} is unavailable.`,
      };
      clearFrame(camera, text[phase] || "Choose and load a world to view the camera.");
      return;
    }
    const response = await request(endpoint);
    const blob = response.status === 204 ? null : await response.blob();
    if (state.session !== session || state.world !== currentWorld || ["loading", "stopping", "offline", "failed"].includes(state.phase)) return;
    if (!blob) {
      clearFrame(camera, `Waiting for the ${name}…`);
      return;
    }
    const previousUrl = image.getAttribute("src");
    image.src = URL.createObjectURL(blob);
    image.hidden = false;
    placeholder.hidden = true;
    if (previousUrl) URL.revokeObjectURL(previousUrl);
  } catch {
    clearFrame(camera, `The ${name} is unavailable.`);
  } finally {
    setTimeout(() => pollFrame(camera), replay ? 750 : 300);
  }
}

async function submit(path, body, onSuccess) {
  if (busy) return;
  busy = true;
  renderState();
  error();
  try {
    await request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    onSuccess?.();
    await updateState();
  } catch (failure) {
    error(failure.message);
  } finally {
    busy = false;
    renderState();
    if (state.phase === "reply") message.focus();
  }
}

byId("world-form").addEventListener("submit", (event) => {
  event.preventDefault();
  submit("/api/world", { world: world.value }, () => {
    cases.selectedIndex = 0;
    message.value = "";
  });
});

stopWorld.addEventListener("click", () => submit("/api/world/stop", {}));

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = message.value.trim();
  if (input) submit("/api/message", { message: input }, () => {
    message.value = "";
    cases.selectedIndex = 0;
  });
});

world.addEventListener("change", renderState);
message.addEventListener("input", renderState);
message.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!byId("send").disabled) byId("chat-form").requestSubmit();
  }
});
cases.addEventListener("change", () => {
  if (cases.value) message.value = cases.value;
  renderState();
  message.focus();
});
logFilter.addEventListener("change", () => {
  for (const [source, rows] of Object.entries(logRows)) {
    for (const row of rows) row.hidden = !logVisible(source);
  }
  logs.scrollTop = logs.scrollHeight;
});
byId("close-image").addEventListener("click", () => dialog.close());

async function start() {
  try {
    const response = await request("/api/options");
    const options = await response.json();
    replay = options.replay;
    if (replay) byId("wrist-title").textContent = "Recorded wrist camera";
    byId("overview-panel").hidden = !options.overview;
    for (const item of options.worlds) {
      worldOptions.set(item.id, item);
      world.add(new Option(item.label, item.id));
    }
    cases.append(matchingCase);
    const groups = new Map();
    for (const item of options.instructions) {
      if (!groups.has(item.group)) {
        const group = document.createElement("optgroup");
        group.label = item.group;
        groups.set(item.group, group);
        cases.append(group);
      }
      groups.get(item.group).append(new Option(item.text, item.text));
    }
    initialized = true;
    pollState();
    pollFrame(cameras.wrist);
    if (options.overview) pollFrame(cameras.overview);
  } catch (failure) {
    error(failure.message);
    state.phase = "offline";
    renderState();
  }
}

start();
