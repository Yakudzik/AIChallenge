from __future__ import annotations

import logging
import os
import uuid

import html

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import HF_MODEL_ID, HF_MODEL_MAGNUM_ID, HF_MODEL_TLAMA_ID
from web_logic import TEMPERATURE_KEY, normalize_temperatures, process_text

_log_level = os.getenv("PYTHONLOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("ai_client").setLevel(_log_level)


app = FastAPI()
_SESSIONS: dict[str, dict[str, dict]] = {}


class MessageIn(BaseModel):
    session_id: str | None = None
    text: str
    temperatures: dict[str, float] | None = None
    provider: str | None = None


def _get_session(session_id: str | None) -> tuple[str, dict, dict]:
    if not session_id:
        session_id = str(uuid.uuid4())
    session = _SESSIONS.setdefault(session_id, {"user_data": {}, "chat_data": {}})
    return session_id, session["user_data"], session["chat_data"]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    hf_model_label = html.escape((HF_MODEL_ID or "Hugging Face").split("/")[-1])
    hf_magnum_label = html.escape((HF_MODEL_MAGNUM_ID or "Magnum").split("/")[-1])
    hf_tinyllama_label = html.escape((HF_MODEL_TLAMA_ID or "TinyLlama").split("/")[-1])
    return """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>White Rabbitt 🐰</title>
    <style>
      @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap");

      :root {
        --bg: #0f141a;
        --panel: #161f2a;
        --accent: #efb65b;
        --text: #f2f1ec;
        --muted: #9aa3ad;
        --shadow: rgba(0, 0, 0, 0.4);
        --layout-gap: 18px;
        --page-padding-bottom: 48px;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        background: radial-gradient(circle at top, #273142 0%, #0f141a 55%, #0b1117 100%);
        color: var(--text);
        min-height: 100vh;
        display: flex;
        justify-content: flex-start;
        padding: var(--layout-gap) 12px var(--page-padding-bottom) 12px;
        overflow: hidden;
      }

      .app {
        width: calc(100vw - 24px);
        display: grid;
        grid-template-columns: minmax(200px, 1fr) minmax(0, 5fr);
        gap: var(--layout-gap);
        height: calc(100vh - var(--layout-gap) - var(--page-padding-bottom));
        box-sizing: border-box;
      }

      .header {
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-width: 0;
      }

      .title {
        font-size: clamp(26px, 4vw, 38px);
        font-weight: 700;
        letter-spacing: 0.5px;
      }

      .subtitle {
        position: relative;
        color: var(--muted);
        font-size: 15px;
      }

      .subtitle-text {
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .tooltip {
        position: absolute;
        left: 0;
        top: 100%;
        margin-top: 8px;
        background: rgba(8, 12, 18, 0.95);
        color: var(--text);
        padding: 8px 10px;
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.4;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35);
        max-width: 520px;
        max-height: min(300px, 80vh);
        overflow: auto;
        width: max-content;
        opacity: 0;
        pointer-events: none;
        transform: translateY(-4px);
        transition: opacity 0.15s ease, transform 0.15s ease;
        z-index: 5;
        white-space: normal;
      }

      .tooltip.visible {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
      }

      .panel {
        background: linear-gradient(140deg, #1a2431 0%, #141d27 100%);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 18px 35px var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.06);
      }

      .messages {
        display: grid;
        gap: 12px;
        overflow-y: auto;
        padding-right: 4px;
        flex: 1;
        min-height: 0;
        scrollbar-gutter: stable;
        height: 100%;
      }

      .messages::-webkit-scrollbar {
        width: 6px;
      }

      .messages::-webkit-scrollbar-thumb {
        background: rgba(239, 182, 91, 0.6);
        border-radius: 999px;
      }

      .message {
        padding: 14px 16px;
        border-radius: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .message.user {
        background: rgba(239, 182, 91, 0.15);
        border: 1px solid rgba(239, 182, 91, 0.4);
        align-self: flex-end;
      }

      .message.bot {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }

      .controls {
        display: grid;
        gap: 12px;
      }

      .sidebar {
        position: sticky;
        top: 24px;
        height: calc(100vh - 56px);
        padding: var(--layout-gap);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(22, 31, 42, 0.95), rgba(14, 20, 28, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 18px 35px var(--shadow);
        display: flex;
        flex-direction: column;
      }

      .sidebar h2 {
        margin: 0;
        font-size: 18px;
        letter-spacing: 0.4px;
      }

      .sidebar-actions {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 16px;
        height: calc(100% - 34px);
        overflow-y: auto;
        padding-right: 4px;
        scrollbar-gutter: stable;
      }

      .sidebar-actions::-webkit-scrollbar {
        width: 6px;
      }

      .sidebar-actions::-webkit-scrollbar-thumb {
        background: rgba(239, 182, 91, 0.4);
        border-radius: 999px;
      }

      .sidebar-actions .action-group {
        width: 100%;
      }

      .sidebar-actions .action-group button {
        flex: 1 1 auto;
      }

      .sidebar-actions .action-group:last-child {
        margin-top: auto;
      }

      .main {
        display: grid;
        grid-template-rows: auto 1fr auto;
        gap: 16px;
        padding-top: var(--layout-gap);
        height: 100%;
      }

      .messages-panel {
        display: flex;
        flex-direction: column;
        min-height: 0;
        flex: 1;
        overflow: hidden;
        height: 460px;
      }

      .input-row {
        display: flex;
        gap: 12px;
        align-items: flex-start;
      }

      textarea {
        width: 100%;
        min-height: 110px;
        resize: vertical;
        background: #0b1117;
        color: var(--text);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
        font-family: "IBM Plex Sans", sans-serif;
        font-size: 15px;
      }

      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
      }

      .action-group {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(15, 20, 26, 0.35);
      }

      .action-group.discussion {
        border-color: rgba(116, 190, 122, 0.9);
        box-shadow: 0 0 0 1px rgba(116, 190, 122, 0.2);
      }

      .action-group.json {
        border-color: rgba(87, 155, 255, 0.9);
        box-shadow: 0 0 0 1px rgba(87, 155, 255, 0.2);
      }

      .action-group.provider {
        border-color: rgba(255, 255, 255, 0.4);
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08);
        gap: 12px;
      }

      .provider-item {
        display: grid;
        gap: 8px;
        flex: 1 1 0;
        min-width: 150px;
      }

      .provider-slider {
        display: grid;
        gap: 6px;
        width: 100%;
      }

      .provider-slider label {
        font-size: 12px;
        color: var(--muted);
      }

      .provider-slider input[type="range"] {
        width: 100%;
        accent-color: var(--accent);
      }

      button {
        border: none;
        background: var(--accent);
        color: #0b1117;
        padding: 10px 18px;
        border-radius: 999px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }

      button.secondary {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: var(--text);
      }

      .provider-button {
        position: relative;
        padding-right: 28px;
      }

      .provider-button.has-chat::after {
        content: "";
        position: absolute;
        right: 12px;
        top: 50%;
        width: 8px;
        height: 8px;
        background: #f28c28;
        border-radius: 50%;
        transform: translateY(-50%);
        box-shadow: 0 0 0 2px rgba(15, 20, 26, 0.8);
      }

      button.secondary.active {
        background: var(--accent);
        color: #0b1117;
        border-color: var(--accent);
      }

      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
      }

      .hint {
        font-family: "IBM Plex Mono", monospace;
        color: var(--muted);
        font-size: 12px;
      }

      .modal {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        background: rgba(9, 12, 16, 0.72);
        z-index: 20;
        padding: 16px;
      }

      .modal.active {
        display: flex;
      }

      .modal-card {
        width: min(720px, 100%);
        background: linear-gradient(160deg, #1b2532 0%, #141d27 100%);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
      }

      .modal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
      }

      .modal-title {
        font-size: 18px;
        font-weight: 600;
      }

      .modal-close {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: var(--text);
        width: 34px;
        height: 34px;
        border-radius: 999px;
        font-size: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
      }

      .modal textarea {
        min-height: 240px;
        width: 100%;
        resize: vertical;
      }

      .prompt-editor {
        display: grid;
        grid-template-columns: minmax(160px, 220px) minmax(0, 1fr);
        gap: 16px;
        align-items: start;
      }

      .prompt-options {
        display: grid;
        gap: 10px;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(11, 17, 23, 0.45);
      }

      .prompt-options-title {
        font-size: 12px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        color: var(--muted);
      }

      .prompt-option {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 14px;
      }

      .prompt-option input {
        accent-color: var(--accent);
      }

      .modal-actions {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
        margin-top: 12px;
      }

      @media (max-width: 720px) {
        .app {
          grid-template-columns: 1fr;
        }
        .sidebar {
          position: static;
          height: auto;
        }
        .panel {
          padding: 16px;
        }
        textarea {
          min-height: 90px;
        }
        .input-row {
          flex-direction: column;
        }
        .prompt-editor {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class="app">
      <aside class="sidebar">
        <h2>Chat settings</h2>
        <div class="sidebar-actions">
          <div class="action-group discussion">
            <button id="discussionToggle" class="secondary" type="button">Discussion: off</button>
          </div>
          <div class="action-group json">
            <button id="jsonToggle" class="secondary" type="button">JSON: on</button>
          </div>
          <div class="action-group">
            <button id="systemPrompt" class="secondary" type="button">System prompt</button>
          </div>
          <div class="action-group provider">
            <div class="provider-item">
              <button id="providerDeepseek" class="secondary provider-button" type="button">Deepseek</button>
              <div class="provider-slider">
                <label for="temperatureDeepseek">Temperature (0-2): <span id="temperatureValueDeepseek">0.5</span></label>
                <input id="temperatureDeepseek" type="range" min="0" max="2" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerHuggingFace" class="secondary provider-button" type="button">""" + hf_model_label + """</button>
              <div class="provider-slider">
                <label for="temperatureHuggingFace">Temperature (0-2): <span id="temperatureValueHuggingFace">0.5</span></label>
                <input id="temperatureHuggingFace" type="range" min="0" max="2" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerHuggingFaceMagnum" class="secondary provider-button" type="button">""" + hf_magnum_label + """</button>
              <div class="provider-slider">
                <label for="temperatureHuggingFaceMagnum">Temperature (0-2): <span id="temperatureValueHuggingFaceMagnum">0.5</span></label>
                <input id="temperatureHuggingFaceMagnum" type="range" min="0" max="2" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerHuggingFaceTinyLlama" class="secondary provider-button" type="button">""" + hf_tinyllama_label + """</button>
              <div class="provider-slider">
                <label for="temperatureHuggingFaceTinyLlama">Temperature (0-2): <span id="temperatureValueHuggingFaceTinyLlama">0.5</span></label>
                <input id="temperatureHuggingFaceTinyLlama" type="range" min="0" max="2" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerYandex" class="secondary provider-button" type="button">Yandex Cloud</button>
              <div class="provider-slider">
                <label for="temperatureYandex">Temperature (0-1): <span id="temperatureValueYandex">0.5</span></label>
                <input id="temperatureYandex" type="range" min="0" max="1" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerClaude" class="secondary provider-button" type="button">Claude</button>
              <div class="provider-slider">
                <label for="temperatureClaude">Temperature (0-1): <span id="temperatureValueClaude">0.5</span></label>
                <input id="temperatureClaude" type="range" min="0" max="1" step="0.1" value="0.5" />
              </div>
            </div>
          </div>
          <div class="action-group">
            <button id="clear" class="secondary" type="button">Сбросить чат</button>
          </div>
        </div>
      </aside>

      <div class="main">
        <header class="header">
          <div class="title">White Rabbit 🐰</div>
          <div id="systemPromptSubtitle" class="subtitle">
            <span id="systemPromptSubtitleText" class="subtitle-text">AI Advent with Love</span>
            <div id="systemPromptTooltip" class="tooltip" role="tooltip"></div>
          </div>
        </header>

        <section class="panel messages-panel">
          <div id="messages" class="messages"></div>
        </section>

        <section class="panel controls">
          <div class="input-row">
            <textarea id="input" placeholder="Введите сообщение..."></textarea>
            <button id="send" type="button">Отправить</button>
          </div>
          <div class="actions"></div>
          <div class="hint">Сессия хранится в браузере до перезагрузки страницы.</div>
        </section>
      </div>
    </main>

      <div id="systemPromptModal" class="modal" role="dialog" aria-modal="true">
      <div class="modal-card">
        <div class="modal-header">
          <div class="modal-title">System prompt</div>
          <button id="systemPromptClose" class="modal-close" type="button" aria-label="Close">×</button>
        </div>
        <div class="prompt-editor">
          <div class="prompt-options">
            <div class="prompt-options-title">Тип промта</div>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="simple" />
              Простой промт
            </label>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="assistant" checked />
              Промт ассистент
            </label>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="mathematician" />
              Математик
            </label>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="philosopher" />
              Гуманитарий
            </label>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="creative" />
              Креативщик
            </label>
            <label class="prompt-option">
              <input type="radio" name="promptType" value="custom" />
              Свой
            </label>
          </div>
          <textarea id="systemPromptText" title="System prompt" placeholder="System prompt"></textarea>
        </div>
        <div class="modal-actions">
          <button id="systemPromptSave" type="button">Сохранить промт</button>
        </div>
      </div>
    </div>

    <script>
      var messagesEl = document.getElementById("messages");
      var inputEl = document.getElementById("input");
      var sendBtn = document.getElementById("send");
      var discussionToggleBtn = document.getElementById("discussionToggle");
      var jsonToggleBtn = document.getElementById("jsonToggle");
      var systemPromptBtn = document.getElementById("systemPrompt");
      var systemPromptSubtitle = document.getElementById("systemPromptSubtitle");
      var systemPromptSubtitleText = document.getElementById("systemPromptSubtitleText");
      var systemPromptTooltip = document.getElementById("systemPromptTooltip");
      var providerDeepseekBtn = document.getElementById("providerDeepseek");
      var providerHuggingFaceBtn = document.getElementById("providerHuggingFace");
      var providerHuggingFaceMagnumBtn = document.getElementById("providerHuggingFaceMagnum");
      var providerHuggingFaceTinyLlamaBtn = document.getElementById("providerHuggingFaceTinyLlama");
      var providerYandexBtn = document.getElementById("providerYandex");
      var providerClaudeBtn = document.getElementById("providerClaude");
      var temperatureDeepseek = document.getElementById("temperatureDeepseek");
      var temperatureValueDeepseek = document.getElementById("temperatureValueDeepseek");
      var temperatureHuggingFace = document.getElementById("temperatureHuggingFace");
      var temperatureValueHuggingFace = document.getElementById("temperatureValueHuggingFace");
      var temperatureHuggingFaceMagnum = document.getElementById("temperatureHuggingFaceMagnum");
      var temperatureValueHuggingFaceMagnum = document.getElementById("temperatureValueHuggingFaceMagnum");
      var temperatureHuggingFaceTinyLlama = document.getElementById("temperatureHuggingFaceTinyLlama");
      var temperatureValueHuggingFaceTinyLlama = document.getElementById("temperatureValueHuggingFaceTinyLlama");
      var temperatureYandex = document.getElementById("temperatureYandex");
      var temperatureValueYandex = document.getElementById("temperatureValueYandex");
      var temperatureClaude = document.getElementById("temperatureClaude");
      var temperatureValueClaude = document.getElementById("temperatureValueClaude");
      var clearBtn = document.getElementById("clear");
      var systemPromptModal = document.getElementById("systemPromptModal");
      var systemPromptClose = document.getElementById("systemPromptClose");
      var systemPromptText = document.getElementById("systemPromptText");
      var systemPromptSave = document.getElementById("systemPromptSave");
      var promptTypeInputs = document.querySelectorAll("input[name='promptType']");

      var SESSION_KEY = "group-experts-session";
      var SESSION_MAP_KEY = "group-experts-sessions";
      var CHAT_MAP_KEY = "group-experts-chats";
      var PROVIDER_KEY = "group-experts-provider";
      var JSON_MODE_MAP_KEY = "group-experts-json-modes";
      var TEMP_MAP_KEY = "group-experts-temps";
      var DISCUSSION_KEY = "group-experts-discussion";
      var sessionId = null;
      var currentProvider = null;
      var sessionsByProvider = {};
      var chatsByProvider = {};
      var jsonModesByProvider = {};
      var tempsByProvider = {};
      var promptTemplates = {
        simple: "",
        assistant: "",
        mathematician: "",
        philosopher: "",
        creative: "",
      };
      var promptTemplatesLoaded = false;
      function generateSessionId() {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
          return window.crypto.randomUUID();
        }
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
          var r = Math.random() * 16 | 0;
          var v = c === "x" ? r : (r & 0x3 | 0x8);
          return v.toString(16);
        });
      }

      function loadStoredJson(key, fallback) {
        var raw = localStorage.getItem(key);
        if (!raw) return fallback;
        try {
          var parsed = JSON.parse(raw);
          return parsed || fallback;
        } catch (error) {
          return fallback;
        }
      }

      function saveStoredJson(key, value) {
        localStorage.setItem(key, JSON.stringify(value));
      }

      function createMessage(text, role) {
        var item = document.createElement("div");
        item.className = "message " + role;
        item.textContent = text;
        return item;
      }

      function appendMessage(text, role, persist) {
        var item = createMessage(text, role);
        messagesEl.appendChild(item);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        if (!persist) return;
        if (!currentProvider) return;
        chatsByProvider[currentProvider] = chatsByProvider[currentProvider] || [];
        chatsByProvider[currentProvider].push({ text: text, role: role });
        saveStoredJson(CHAT_MAP_KEY, chatsByProvider);
        updateProviderIndicators();
      }

      function addMessage(text, role) {
        appendMessage(text, role, true);
      }

      var discussionEnabled = false;
      function setDiscussionState(isOn) {
        discussionEnabled = Boolean(isOn);
        discussionToggleBtn.classList.toggle("active", discussionEnabled);
        discussionToggleBtn.textContent = discussionEnabled ? "Discussion: on" : "Discussion: off";
        localStorage.setItem(DISCUSSION_KEY, discussionEnabled ? "on" : "off");
      }

      var jsonEnabled = true;
      function setJsonState(mode) {
        jsonEnabled = mode === "on";
        jsonToggleBtn.classList.toggle("active", jsonEnabled);
        jsonToggleBtn.textContent = jsonEnabled ? "JSON: on" : "JSON: off";
      }

      function setProviderState(mode) {
        providerDeepseekBtn.classList.toggle("active", mode === "deepseek");
        providerHuggingFaceBtn.classList.toggle("active", mode === "huggingface");
        providerHuggingFaceMagnumBtn.classList.toggle("active", mode === "huggingface-magnum");
        providerHuggingFaceTinyLlamaBtn.classList.toggle("active", mode === "huggingface-tinyllama");
        providerYandexBtn.classList.toggle("active", mode === "yandex");
        providerClaudeBtn.classList.toggle("active", mode === "claude");
      }

      function openSystemPromptModal(text) {
        systemPromptText.value = text || "";
        systemPromptModal.classList.add("active");
      }

      function closeSystemPromptModal() {
        systemPromptModal.classList.remove("active");
      }

      function saveSystemPrompt() {
        var text = systemPromptText.value || "";
        requestCommand("/set_system_prompt" + text)
          .then(function (response) {
            if (response) {
              addMessage(response, "bot");
            }
            refreshSystemPromptSubtitle();
            closeSystemPromptModal();
          })
          .catch(function (error) {
            addMessage("Ошибка сохранения промта: " + error, "bot");
          });
      }

      function send(text) {
        if (!text.trim()) return Promise.resolve();
        addMessage(text, "user");
        inputEl.value = "";
        return fetch("/api/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            text: text,
            temperatures: getTemperatures(),
            provider: currentProvider,
          }),
        })
          .then(function (response) { return response.json(); })
          .then(function (data) {
            if (data.session_id && data.session_id !== sessionId) {
              sessionId = data.session_id;
              localStorage.setItem(SESSION_KEY, sessionId);
            }
            (data.messages || []).forEach(function (msg) { addMessage(msg, "bot"); });
          })
          .catch(function (error) {
            addMessage("Ошибка: " + error, "bot");
          });
      }

      function requestCommand(command) {
        return fetch("/api/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            text: command,
            temperatures: getTemperatures(),
            provider: currentProvider,
          }),
        })
          .then(function (response) { return response.json(); })
          .then(function (data) {
            if (data.session_id && data.session_id !== sessionId) {
              sessionId = data.session_id;
              localStorage.setItem(SESSION_KEY, sessionId);
            }
            return (data.messages || []).join("").trim();
          });
      }

      function setSystemPromptSubtitle(text) {
        if (!systemPromptSubtitleText) return;
        systemPromptSubtitleText.textContent = text || "";
        if (systemPromptTooltip) {
          systemPromptTooltip.textContent = text || "";
        }
      }

      function refreshSystemPromptSubtitle() {
        return requestCommand("/system_prompt")
          .then(function (response) {
            if (typeof response === "string") {
              setSystemPromptSubtitle(response);
            }
          })
          .catch(function () {});
      }

      var tooltipTimer = null;
      var tooltipHideTimer = null;
      function showSystemPromptTooltip() {
        if (!systemPromptTooltip) return;
        systemPromptTooltip.classList.add("visible");
      }

      function hideSystemPromptTooltip() {
        if (!systemPromptTooltip) return;
        systemPromptTooltip.classList.remove("visible");
      }

      if (systemPromptSubtitle && systemPromptTooltip) {
        systemPromptSubtitle.addEventListener("mouseenter", function () {
          if (tooltipTimer) {
            clearTimeout(tooltipTimer);
          }
          if (tooltipHideTimer) {
            clearTimeout(tooltipHideTimer);
            tooltipHideTimer = null;
          }
          tooltipTimer = setTimeout(showSystemPromptTooltip, 2000);
        });
        systemPromptSubtitle.addEventListener("mouseleave", function () {
          if (tooltipTimer) {
            clearTimeout(tooltipTimer);
            tooltipTimer = null;
          }
          tooltipHideTimer = setTimeout(hideSystemPromptTooltip, 150);
        });
        systemPromptTooltip.addEventListener("mouseenter", function () {
          if (tooltipTimer) {
            clearTimeout(tooltipTimer);
            tooltipTimer = null;
          }
          if (tooltipHideTimer) {
            clearTimeout(tooltipHideTimer);
            tooltipHideTimer = null;
          }
          showSystemPromptTooltip();
        });
        systemPromptTooltip.addEventListener("mouseleave", function () {
          if (tooltipHideTimer) {
            clearTimeout(tooltipHideTimer);
          }
          tooltipHideTimer = setTimeout(hideSystemPromptTooltip, 150);
        });
      }

      function ensurePromptTemplates() {
        if (promptTemplatesLoaded) {
          return Promise.resolve();
        }
        return requestCommand("/prompt_templates")
          .then(function (response) {
            if (!response) {
              promptTemplatesLoaded = true;
              return;
            }
            try {
              var parsed = JSON.parse(response);
              promptTemplates = Object.assign(promptTemplates, parsed);
            } catch (error) {
              addMessage("Ошибка чтения шаблонов промта: " + error, "bot");
            }
            promptTemplatesLoaded = true;
          })
          .catch(function () {
            promptTemplatesLoaded = true;
          });
      }

      function applyPromptType(type) {
        if (type === "simple") {
          systemPromptText.value = promptTemplates.simple || "";
          return;
        }
        if (type === "assistant") {
          systemPromptText.value = promptTemplates.assistant || "";
          return;
        }
        if (type === "mathematician") {
          systemPromptText.value = promptTemplates.mathematician || "";
          return;
        }
        if (type === "philosopher") {
          systemPromptText.value = promptTemplates.philosopher || "";
          return;
        }
        if (type === "creative") {
          systemPromptText.value = promptTemplates.creative || "";
          return;
        }
        systemPromptText.value = "Напишите свой промт";
      }

      function setPromptType(type) {
        promptTypeInputs.forEach(function (inputEl) {
          inputEl.checked = inputEl.value === type;
        });
        applyPromptType(type);
      }

      sendBtn.addEventListener("click", function () { send(inputEl.value); });
      inputEl.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          send(inputEl.value);
        }
      });

      discussionToggleBtn.addEventListener("click", function () {
        var nextState = !discussionEnabled;
        setDiscussionState(nextState);
        send(nextState ? "/discussion_on" : "/discussion_off");
      });
      jsonToggleBtn.addEventListener("click", function () {
        var nextMode = jsonEnabled ? "off" : "on";
        setJsonState(nextMode);
        if (currentProvider) {
          jsonModesByProvider[currentProvider] = nextMode;
          saveStoredJson(JSON_MODE_MAP_KEY, jsonModesByProvider);
        }
        send(nextMode === "on" ? "/json_on" : "/json_off");
      });
      systemPromptBtn.addEventListener("click", function () {
        ensurePromptTemplates().then(function () {
          setPromptType("assistant");
          openSystemPromptModal(systemPromptText.value);
        });
      });
      systemPromptClose.addEventListener("click", closeSystemPromptModal);
      systemPromptSave.addEventListener("click", saveSystemPrompt);
      systemPromptModal.addEventListener("click", function (event) {
        if (event.target === systemPromptModal) {
          closeSystemPromptModal();
        }
      });
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
          closeSystemPromptModal();
        }
      });
      promptTypeInputs.forEach(function (inputEl) {
        inputEl.addEventListener("change", function () {
          if (inputEl.checked) {
            applyPromptType(inputEl.value);
          }
        });
      });
      providerDeepseekBtn.addEventListener("click", function () {
        switchProvider("deepseek", "/use_deepseek");
      });
      providerHuggingFaceBtn.addEventListener("click", function () {
        switchProvider("huggingface", "/use_huggingface");
      });
      providerHuggingFaceMagnumBtn.addEventListener("click", function () {
        switchProvider("huggingface-magnum", "/use_huggingface_magnum");
      });
      providerHuggingFaceTinyLlamaBtn.addEventListener("click", function () {
        switchProvider("huggingface-tinyllama", "/use_huggingface_tinyllama");
      });
      providerYandexBtn.addEventListener("click", function () {
        switchProvider("yandex", "/use_yandex");
      });
      providerClaudeBtn.addEventListener("click", function () {
        switchProvider("claude", "/use_claude");
      });
      var temperatureRanges = {
        deepseek: { min: 0, max: 2 },
        huggingface: { min: 0, max: 2 },
        "huggingface-magnum": { min: 0, max: 2 },
        "huggingface-tinyllama": { min: 0, max: 2 },
        yandex: { min: 0, max: 1 },
        claude: { min: 0, max: 1 },
      };

      function clampTemperature(provider, value) {
        var range = temperatureRanges[provider];
        if (!range) return value;
        var numeric = Number(value);
        if (Number.isNaN(numeric)) {
          return range.min;
        }
        return Math.min(range.max, Math.max(range.min, numeric));
      }

      function bindTemperatureWithStore(provider, inputEl, valueEl) {
        if (!inputEl || !valueEl) return;
        function persistValue() {
          if (!provider) return;
          tempsByProvider[provider] = inputEl.value;
          saveStoredJson(TEMP_MAP_KEY, tempsByProvider);
        }
        inputEl.addEventListener("input", function () {
          valueEl.textContent = inputEl.value;
          persistValue();
        });
        inputEl.addEventListener("change", function () {
          valueEl.textContent = inputEl.value;
          persistValue();
        });
        valueEl.textContent = inputEl.value;
      }

      bindTemperatureWithStore("deepseek", temperatureDeepseek, temperatureValueDeepseek);
      bindTemperatureWithStore("huggingface", temperatureHuggingFace, temperatureValueHuggingFace);
      bindTemperatureWithStore("huggingface-magnum", temperatureHuggingFaceMagnum, temperatureValueHuggingFaceMagnum);
      bindTemperatureWithStore("huggingface-tinyllama", temperatureHuggingFaceTinyLlama, temperatureValueHuggingFaceTinyLlama);
      bindTemperatureWithStore("yandex", temperatureYandex, temperatureValueYandex);
      bindTemperatureWithStore("claude", temperatureClaude, temperatureValueClaude);

      function setTemperature(inputEl, valueEl, value) {
        if (!inputEl || !valueEl) return;
        inputEl.value = value;
        valueEl.textContent = value;
      }

      function getTemperatures() {
        return {
          deepseek: clampTemperature(
            "deepseek",
            temperatureDeepseek ? temperatureDeepseek.value : "0.6"
          ),
          huggingface: clampTemperature(
            "huggingface",
            temperatureHuggingFace ? temperatureHuggingFace.value : "0.6"
          ),
          "huggingface-magnum": clampTemperature(
            "huggingface-magnum",
            temperatureHuggingFaceMagnum ? temperatureHuggingFaceMagnum.value : "0.6"
          ),
          "huggingface-tinyllama": clampTemperature(
            "huggingface-tinyllama",
            temperatureHuggingFaceTinyLlama ? temperatureHuggingFaceTinyLlama.value : "0.6"
          ),
          yandex: clampTemperature(
            "yandex",
            temperatureYandex ? temperatureYandex.value : "0.6"
          ),
          claude: clampTemperature(
            "claude",
            temperatureClaude ? temperatureClaude.value : "0.6"
          ),
        };
      }

      function resetUiState() {
        if (currentProvider) {
          setProviderState(currentProvider);
        }
      }

      function getProviderButtons() {
        return {
          deepseek: providerDeepseekBtn,
          huggingface: providerHuggingFaceBtn,
          "huggingface-magnum": providerHuggingFaceMagnumBtn,
          "huggingface-tinyllama": providerHuggingFaceTinyLlamaBtn,
          yandex: providerYandexBtn,
          claude: providerClaudeBtn,
        };
      }

      function updateProviderIndicators() {
        var buttons = getProviderButtons();
        Object.keys(buttons).forEach(function (provider) {
          var btn = buttons[provider];
          if (!btn) return;
          var hasChat = (chatsByProvider[provider] || []).length > 0;
          btn.classList.toggle("has-chat", hasChat);
        });
      }

      function renderMessages(provider) {
        messagesEl.innerHTML = "";
        var cached = chatsByProvider[provider] || [];
        cached.forEach(function (item) {
          if (!item || !item.text) return;
          appendMessage(item.text, item.role || "bot", false);
        });
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      function applyJsonMode(provider) {
        var savedMode = jsonModesByProvider[provider];
        var mode = savedMode === "off" ? "off" : "on";
        setJsonState(mode);
        if (savedMode) {
          requestCommand(mode === "on" ? "/json_on" : "/json_off").catch(function () {});
        }
      }

      function applyTemperatures(provider) {
        var defaults = {
          deepseek: "0.5",
          huggingface: "0.5",
          "huggingface-magnum": "0.5",
          "huggingface-tinyllama": "0.5",
          yandex: "0.5",
          claude: "0.5",
        };
        var value = typeof tempsByProvider[provider] === "string"
          ? tempsByProvider[provider]
          : defaults[provider] || "0.5";
        if (provider === "deepseek") {
          setTemperature(temperatureDeepseek, temperatureValueDeepseek, value);
        }
        if (provider === "huggingface") {
          setTemperature(temperatureHuggingFace, temperatureValueHuggingFace, value);
        }
        if (provider === "huggingface-magnum") {
          setTemperature(temperatureHuggingFaceMagnum, temperatureValueHuggingFaceMagnum, value);
        }
        if (provider === "huggingface-tinyllama") {
          setTemperature(temperatureHuggingFaceTinyLlama, temperatureValueHuggingFaceTinyLlama, value);
        }
        if (provider === "yandex") {
          setTemperature(temperatureYandex, temperatureValueYandex, value);
        }
        if (provider === "claude") {
          setTemperature(temperatureClaude, temperatureValueClaude, value);
        }
      }

      function ensureProviderSettings(provider) {
        applyJsonMode(provider);
        applyTemperatures(provider);
      }

      function ensureSessionForProvider(provider) {
        if (!provider) return generateSessionId();
        var existing = sessionsByProvider[provider];
        if (existing) return existing;
        var newId = generateSessionId();
        sessionsByProvider[provider] = newId;
        saveStoredJson(SESSION_MAP_KEY, sessionsByProvider);
        return newId;
      }

      function switchProvider(provider, command) {
        if (!provider) return;
        currentProvider = provider;
        localStorage.setItem(PROVIDER_KEY, currentProvider);
        sessionId = ensureSessionForProvider(provider);
        localStorage.setItem(SESSION_KEY, sessionId);
        setProviderState(provider);
        renderMessages(provider);
        updateProviderIndicators();
        ensureProviderSettings(provider);
        inputEl.value = "";
        if (discussionEnabled) {
          requestCommand("/discussion_on").catch(function () {});
        }
        if (command) {
          requestCommand(command).then(refreshSystemPromptSubtitle);
        } else {
          refreshSystemPromptSubtitle();
        }
      }

      clearBtn.addEventListener("click", function () {
        requestCommand("/reset_chat")
          .then(function () {
            sessionId = generateSessionId();
            if (currentProvider) {
              sessionsByProvider[currentProvider] = sessionId;
              saveStoredJson(SESSION_MAP_KEY, sessionsByProvider);
              chatsByProvider[currentProvider] = [];
              saveStoredJson(CHAT_MAP_KEY, chatsByProvider);
              jsonModesByProvider[currentProvider] = "on";
              saveStoredJson(JSON_MODE_MAP_KEY, jsonModesByProvider);
              tempsByProvider[currentProvider] = "0.5";
              saveStoredJson(TEMP_MAP_KEY, tempsByProvider);
            }
            localStorage.setItem(SESSION_KEY, sessionId);
            messagesEl.innerHTML = "";
            inputEl.value = "";
            closeSystemPromptModal();
            resetUiState();
            updateProviderIndicators();
            ensureProviderSettings(currentProvider);
            if (discussionEnabled) {
              requestCommand("/discussion_on").catch(function () {});
            }
            refreshSystemPromptSubtitle();
          })
          .catch(function (error) {
            addMessage("Ошибка сброса: " + error, "bot");
          });
      });

      sessionsByProvider = loadStoredJson(SESSION_MAP_KEY, {});
      chatsByProvider = loadStoredJson(CHAT_MAP_KEY, {});
      jsonModesByProvider = loadStoredJson(JSON_MODE_MAP_KEY, {});
      tempsByProvider = loadStoredJson(TEMP_MAP_KEY, {});
      discussionEnabled = localStorage.getItem(DISCUSSION_KEY) === "on";
      currentProvider = localStorage.getItem(PROVIDER_KEY) || "deepseek";
      sessionId = sessionsByProvider[currentProvider] || localStorage.getItem(SESSION_KEY);
      if (!sessionId) {
        sessionId = generateSessionId();
      }
      sessionsByProvider[currentProvider] = sessionId;
      saveStoredJson(SESSION_MAP_KEY, sessionsByProvider);
      localStorage.setItem(SESSION_KEY, sessionId);

      resetUiState();
      renderMessages(currentProvider);
      updateProviderIndicators();
      ensureProviderSettings(currentProvider);
      setDiscussionState(discussionEnabled);
      refreshSystemPromptSubtitle();
    </script>
  </body>
  </html>
    """


@app.post("/api/message")
def message(payload: MessageIn):
    session_id, user_data, chat_data = _get_session(payload.session_id)
    if payload.temperatures:
        user_data[TEMPERATURE_KEY] = normalize_temperatures(payload.temperatures)
    if payload.provider:
        user_data["ai_provider"] = payload.provider
    messages = process_text(payload.text, user_data, chat_data)
    return {"session_id": session_id, "messages": messages}
