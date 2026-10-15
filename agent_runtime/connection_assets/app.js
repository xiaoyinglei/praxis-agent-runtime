"use strict";
const fragment = location.hash.slice(1);
if (fragment) { sessionStorage.setItem("praxis-connections-session", fragment); history.replaceState(null, "", "/"); }
const token = sessionStorage.getItem("praxis-connections-session") || "";
const notice = document.querySelector("#notice");
const active = new Set();
let installed = new Set();
function message(text, error = false, authorizationUrl = null) {
  notice.hidden = false; notice.className = error ? "error" : ""; notice.textContent = text;
  if (authorizationUrl) {
    const target = new URL(authorizationUrl);
    if (target.protocol !== "https:") return;
    const link = document.createElement("a"); link.href = target.href; link.target = "_blank";
    link.rel = "noopener noreferrer"; link.textContent = "前往服务商授权 ↗"; notice.append(link);
  }
}
async function api(path, body) {
  const options = {headers: {Authorization: `Bearer ${token}`}, cache: "no-store"};
  if (body !== undefined) { options.method = "POST"; options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "连接管理暂时不可用。");
  return data;
}
function element(tag, text, className) {
  const node = document.createElement(tag); if (text) node.textContent = text; if (className) node.className = className; return node;
}
function button(text, callback, className) {
  const node = element("button", text, className); node.type = "button";
  node.addEventListener("click", callback); return node;
}
async function run(path, body, label) {
  if (active.has(label)) return;
  active.add(label); renderBusy();
  message(`正在连接 ${label}…`);
  try {
    const job = await api(path, body);
    for (;;) {
      const state = await api(`/api/jobs/${job.id}`);
      if (state.status === "error") throw new Error(state.message);
      if (state.status === "done") {
        const result = state.result;
        if (result.removed) message(`已移除 ${label} 的本机连接。`);
        else if (result.connected) message(`${label} 已连接，发现 ${result.tools.length} 个可用工具。新开 Agent 会话即可使用。`);
        else message(`${label} 已添加，但连接检查未通过。请检查认证、服务工具范围和项目权限后重试。`, true);
        break;
      }
      if (state.authorization_url) message(`请在服务商页面完成 ${label} 授权，完成后此页面会自动更新。`, false, state.authorization_url);
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } catch (error) { message(error.message, true); }
  finally { active.delete(label); await refresh(); renderBusy(); }
}
function renderBusy() {
  document.querySelectorAll("button[data-name]").forEach(node => {
    const busy = active.has(node.dataset.name);
    node.disabled = busy; if (node.dataset.connect === "true") node.textContent = busy ? "连接中…" : installed.has(node.dataset.name) ? "已添加" : "连接 ↗";
    if (node.dataset.connect === "true" && installed.has(node.dataset.name)) node.disabled = true;
  });
  document.querySelector("#custom-form button[type=submit]").disabled = active.size > 0;
}
async function refresh() {
  try {
    const connections = await api("/api/connections"); installed = new Set(connections.map(item => item.name));
    const list = document.querySelector("#connections"); list.replaceChildren();
    if (!connections.length) list.append(element("p", "还没有连接。选择上方服务，或填写自己的 MCP 地址。", "muted"));
    for (const entry of connections) {
      const row = element("div", null, "connection"), info = element("div");
      info.append(element("strong", entry.name), element("small", `${entry.url || "本地服务"} · ${entry.read_only === false ? "允许动作" : "只读"}`));
      row.append(info); const actions = element("div", null, "actions");
      if (entry.auth === "oauth") actions.append(button("重新授权", () => run("/api/login", {name: entry.name}, entry.name)));
      actions.append(button("检查连接", () => run("/api/check", {name: entry.name}, entry.name)));
      actions.append(button("移除", () => { if (confirm(`移除 ${entry.name} 的本机连接？服务商侧授权需在服务商设置中撤销。`)) run("/api/remove", {name: entry.name}, entry.name); }));
      actions.querySelectorAll("button").forEach(node => {node.dataset.name = entry.name;});
      row.append(actions); list.append(row);
    }
    renderBusy();
  } catch (error) { message(error.message, true); }
}
async function init() {
  if (!token) {message("请使用 agent connections 本次启动时显示的完整地址打开页面。", true); return;}
  try {
    const services = await api("/api/catalog"), catalog = document.querySelector("#catalog");
    for (const service of services) {
      const card = element("article", null, "card"); card.dataset.id = service.id;
      card.append(element("div", {github: "⌘", notion: "N", linear: "◒"}[service.id] || "+", "icon"));
      card.append(element("h3", service.title), element("p", service.description));
      if (service.detail) card.append(element("p", service.detail, "service-detail"));
      const bottom = element("div", null, "bottom"), docs = element("a", "服务说明 ↗");
      docs.href = service.documentation; docs.target = "_blank"; docs.rel = "noopener noreferrer";
      const connect = button("连接 ↗", () => run("/api/add", {preset: service.id}, service.id));
      connect.dataset.name = service.id; connect.dataset.connect = "true"; bottom.append(docs, connect); card.append(bottom); catalog.append(card);
    }
    await refresh();
  } catch (error) { message(error.message, true); }
}
document.querySelector("#refresh").addEventListener("click", refresh);
document.querySelector("select[name=auth]").addEventListener("change", event => {
  document.querySelector("#token-field").hidden = event.target.value !== "environment";
  document.querySelector("#oauth-options").hidden = event.target.value !== "oauth";
});
document.querySelector("#custom-form").addEventListener("submit", event => {
  event.preventDefault(); const data = new FormData(event.target);
  const body = {name: data.get("name").trim(), url: data.get("url").trim(), auth: data.get("auth"), read_only: !data.has("allow_actions")};
  if (body.auth === "environment") body.bearer_token_env_var = data.get("bearer_token_env_var").trim();
  if (body.auth === "oauth") {
    body.oauth_client_id = data.get("oauth_client_id").trim() || null;
    body.oauth_callback_port = Number(data.get("oauth_callback_port")) || 0;
    body.oauth_scopes = data.get("oauth_scopes").trim().split(/\s+/).filter(Boolean);
    body.oauth_allowed_domains = data.get("oauth_allowed_domains").split(",").map(item => item.trim()).filter(Boolean);
  }
  run("/api/add", body, body.name);
});
init();
