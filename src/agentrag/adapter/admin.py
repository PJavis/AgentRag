"""Admin reasoning panel — LangGraph-style agent trace viewer."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.agentrag.adapter.auth import is_admin
from src.agentrag.chat.history import ConversationStore

router = APIRouter(prefix="/admin")

_ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>AgentRag — Reasoning Inspector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
  header { background: #1e293b; padding: 1rem 1.5rem; border-bottom: 1px solid #334155;
           display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #f1f5f9; }
  header span { font-size: .75rem; color: #64748b; }
  .layout { display: grid; grid-template-columns: 300px 1fr; height: calc(100vh - 57px); }
  .sidebar { background: #1e293b; border-right: 1px solid #334155; overflow-y: auto; }
  .sidebar-item { padding: .75rem 1rem; border-bottom: 1px solid #1e293b; cursor: pointer;
                  transition: background .1s; }
  .sidebar-item:hover { background: #334155; }
  .sidebar-item.active { background: #1d4ed8; }
  .sidebar-item .title { font-size: .85rem; font-weight: 500; white-space: nowrap;
                          overflow: hidden; text-overflow: ellipsis; }
  .sidebar-item .meta { font-size: .7rem; color: #94a3b8; margin-top: .2rem; }
  .main { overflow-y: auto; padding: 1.5rem; }
  .empty { color: #475569; font-size: .9rem; padding: 3rem; text-align: center; }
  .step { background: #1e293b; border: 1px solid #334155; border-radius: .5rem;
          margin-bottom: 1rem; overflow: hidden; }
  .step-header { display: flex; align-items: center; gap: .75rem; padding: .75rem 1rem;
                 background: #0f172a; border-bottom: 1px solid #334155; }
  .step-num { width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #1d4ed8;
              display: flex; align-items: center; justify-content: center;
              font-size: .7rem; font-weight: 700; flex-shrink: 0; }
  .step-name { font-size: .85rem; font-weight: 600; color: #93c5fd; }
  .step-time { font-size: .7rem; color: #64748b; margin-left: auto; }
  .step-body { padding: .75rem 1rem; }
  .label { font-size: .7rem; font-weight: 600; color: #64748b; text-transform: uppercase;
           letter-spacing: .05em; margin-bottom: .3rem; }
  pre { background: #0f172a; border: 1px solid #334155; border-radius: .3rem;
        padding: .6rem .8rem; font-size: .75rem; line-height: 1.5;
        overflow-x: auto; white-space: pre-wrap; word-break: break-word;
        color: #a5f3fc; }
  .flow { display: flex; flex-direction: column; gap: .5rem; margin-bottom: 1.5rem; }
  .flow-node { background: #1e293b; border: 1px solid #334155; border-radius: .4rem;
               padding: .5rem .8rem; font-size: .8rem; position: relative; }
  .flow-node::after { content: "▼"; display: block; text-align: center; color: #334155;
                      font-size: .8rem; margin-top: .3rem; }
  .flow-node:last-child::after { display: none; }
  .flow-node.tool { border-color: #1d4ed8; }
  .flow-node.answer { border-color: #16a34a; }
  .badge { display: inline-block; padding: .15rem .4rem; border-radius: .25rem;
           font-size: .65rem; font-weight: 600; background: #1d4ed8; color: #fff;
           margin-right: .3rem; }
  .badge.green { background: #16a34a; }
  .conv-title { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #f1f5f9; }
  .messages { margin-bottom: 1.5rem; }
  .msg { padding: .6rem .8rem; border-radius: .4rem; margin-bottom: .4rem; font-size: .85rem; }
  .msg.user { background: #1e3a5f; }
  .msg.assistant { background: #1e293b; border: 1px solid #334155; }
  .msg-role { font-size: .7rem; color: #64748b; margin-bottom: .2rem; }
  .section-title { font-size: .8rem; font-weight: 600; color: #64748b;
                   text-transform: uppercase; letter-spacing: .05em;
                   margin: 1.5rem 0 .6rem; }
  #token-input { background: #1e293b; border: 1px solid #334155; color: #e2e8f0;
                 padding: .4rem .6rem; border-radius: .3rem; font-size: .8rem;
                 width: 200px; }
  #token-btn { background: #1d4ed8; color: #fff; border: none; padding: .4rem .8rem;
               border-radius: .3rem; font-size: .8rem; cursor: pointer; margin-left: .4rem; }
</style>
</head>
<body>
<header>
  <h1>🧠 Agent Reasoning Inspector</h1>
  <span>AgentRag Admin</span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:.5rem">
    <input id="token-input" type="password" placeholder="Admin token" />
    <button id="token-btn" onclick="loadConversations()">Load</button>
  </div>
</header>
<div class="layout">
  <div class="sidebar" id="sidebar">
    <div class="empty">Enter admin token →</div>
  </div>
  <div class="main" id="main">
    <div class="empty">Select a conversation to inspect its reasoning trace.</div>
  </div>
</div>
<script>
let adminToken = '';

// Detect base path so this works under any mount prefix (e.g. /on/admin)
const ADMIN_BASE = window.location.pathname.replace(/\/+$/, '').replace(/\/admin$/, '') + '/admin';

function getToken() {
  return document.getElementById('token-input').value.trim();
}

async function apiFetch(path) {
  const t = getToken();
  const r = await fetch(ADMIN_BASE + path, { headers: { 'X-Admin-Token': t } });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

async function loadConversations() {
  adminToken = getToken();
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const data = await apiFetch('/api/conversations');
    if (!data.length) { sidebar.innerHTML = '<div class="empty">No conversations</div>'; return; }
    sidebar.innerHTML = data.map((c, i) => `
      <div class="sidebar-item" onclick="loadTrace('${c.id}', this)">
        <div class="title">${esc(c.title || 'Untitled')}</div>
        <div class="meta">${c.message_count} msgs · ${new Date(c.created_at).toLocaleDateString()}</div>
      </div>`).join('');
  } catch(e) { sidebar.innerHTML = '<div class="empty">Error: ' + e.message + '</div>'; }
}

async function loadTrace(convId, el) {
  document.querySelectorAll('.sidebar-item').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  const main = document.getElementById('main');
  main.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const data = await apiFetch('/api/conversations/' + convId + '/trace');
    renderTrace(main, data);
  } catch(e) { main.innerHTML = '<div class="empty">Error: ' + e.message + '</div>'; }
}

function renderTrace(el, data) {
  const msgs = data.messages || [];
  const traces = data.tool_traces || [];

  let html = `<div class="conv-title">${esc(data.title || 'Untitled conversation')}</div>`;

  // Messages
  html += `<div class="section-title">Conversation</div><div class="messages">`;
  for (const m of msgs) {
    html += `<div class="msg ${m.role}"><div class="msg-role">${m.role}</div>${esc(m.content)}</div>`;
  }
  html += `</div>`;

  // Reasoning trace per assistant message
  if (traces.length) {
    html += `<div class="section-title">Agent Reasoning Trace</div>`;
    traces.forEach((trace, ti) => {
      html += `<div style="margin-bottom:1.5rem">
        <div style="font-size:.75rem;color:#64748b;margin-bottom:.5rem">Turn ${ti+1}</div>`;

      // Flow diagram
      html += `<div class="flow">`;
      html += `<div class="flow-node"><span class="badge">START</span> User question received</div>`;
      (trace || []).forEach(step => {
        html += `<div class="flow-node tool">
          <span class="badge">${esc(step.tool_name || 'step')}</span>${esc(step.tool_input?.query || step.tool_name || '')}
        </div>`;
      });
      html += `<div class="flow-node answer"><span class="badge green">ANSWER</span> Response synthesized</div>`;
      html += `</div>`;

      // Step details
      (trace || []).forEach((step, si) => {
        const dur = step.duration_ms ? ` · ${Math.round(step.duration_ms)}ms` : '';
        html += `<div class="step">
          <div class="step-header">
            <div class="step-num">${si+1}</div>
            <div class="step-name">${esc(step.tool_name || 'step')}</div>
            <div class="step-time">${dur}</div>
          </div>
          <div class="step-body">
            <div class="label">Input</div>
            <pre>${esc(JSON.stringify(step.tool_input, null, 2))}</pre>
            <div class="label" style="margin-top:.6rem">Output summary</div>
            <pre>${esc(truncate(JSON.stringify(step.tool_output || step.output || {}), 500))}</pre>
          </div>
        </div>`;
      });
      html += `</div>`;
    });
  } else {
    html += `<div class="empty" style="padding:1rem">No reasoning trace available for this conversation.</div>`;
  }

  el.innerHTML = html;
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function truncate(s, n) { return s.length > n ? s.slice(0,n) + '…' : s; }
</script>
</body>
</html>"""


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_page():
    return HTMLResponse(_ADMIN_HTML)


@router.get("/api/conversations")
async def list_conversations_admin(request: Request):
    if not is_admin(request):
        raise HTTPException(403, "Admin token required")
    store = ConversationStore()
    convs = await store.list_conversations(limit=100)
    result = []
    for c in convs:
        msgs = await store.list_messages(c["conversation_id"], limit=1000)
        result.append({
            "id": c["conversation_id"],
            "title": c.get("title"),
            "message_count": len(msgs),
            "created_at": c.get("created_at", ""),
        })
    return result


@router.get("/api/conversations/{conversation_id}/trace")
async def get_conversation_trace(conversation_id: str, request: Request):
    if not is_admin(request):
        raise HTTPException(403, "Admin token required")

    store = ConversationStore()
    conv = await store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    msgs = await store.list_messages(conversation_id, limit=1000)
    tool_traces = [
        m["tool_trace"]
        for m in msgs
        if m.get("role") == "assistant" and m.get("tool_trace")
    ]

    return {
        "id": conversation_id,
        "title": conv.get("title"),
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in msgs
        ],
        "tool_traces": tool_traces,
    }
