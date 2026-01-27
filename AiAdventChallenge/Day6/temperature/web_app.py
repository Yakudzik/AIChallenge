from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from web_logic import TEMPERATURE_KEY, normalize_temperatures, process_text


app = FastAPI()
_SESSIONS: dict[str, dict[str, dict]] = {}


class MessageIn(BaseModel):
    session_id: str | None = None
    text: str
    temperatures: dict[str, float] | None = None


def _get_session(session_id: str | None) -> tuple[str, dict, dict]:
    if not session_id:
        session_id = str(uuid.uuid4())
    session = _SESSIONS.setdefault(session_id, {"user_data": {}, "chat_data": {}})
    return session_id, session["user_data"], session["chat_data"]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
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
        grid-template-columns: minmax(180px, 1fr) minmax(0, 5fr);
        gap: var(--layout-gap);
        height: calc(100vh - var(--layout-gap) - var(--page-padding-bottom));
        box-sizing: border-box;
      }

      .header {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .title {
        font-size: clamp(26px, 4vw, 38px);
        font-weight: 700;
        letter-spacing: 0.5px;
      }

      .subtitle {
        color: var(--muted);
        font-size: 15px;
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
        height: 420px;
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
            <button id="discussionOn" class="secondary" type="button">/discussion_on</button>
            <button id="discussionOff" class="secondary" type="button">/discussion_off</button>
          </div>
          <div class="action-group json">
            <button id="jsonOn" class="secondary" type="button">/json_on</button>
            <button id="jsonOff" class="secondary" type="button">/json_off</button>
          </div>
          <div class="action-group">
            <button id="systemPrompt" class="secondary" type="button">System prompt</button>
          </div>
          <div class="action-group provider">
            <div class="provider-item">
              <button id="providerDeepseek" class="secondary" type="button">Deepseek</button>
              <div class="provider-slider">
                <label for="temperatureDeepseek">Temperature (0-2): <span id="temperatureValueDeepseek">0.5</span></label>
                <input id="temperatureDeepseek" type="range" min="0" max="2" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerYandex" class="secondary" type="button">Yandex Cloud</button>
              <div class="provider-slider">
                <label for="temperatureYandex">Temperature (0-1): <span id="temperatureValueYandex">0.5</span></label>
                <input id="temperatureYandex" type="range" min="0" max="1" step="0.1" value="0.5" />
              </div>
            </div>
            <div class="provider-item">
              <button id="providerClaude" class="secondary" type="button">Claude</button>
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
          <div class="title">White Rabbitt 🐰</div>
          <div class="subtitle">AI Advent with Love</div>
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
      var discussionOnBtn = document.getElementById("discussionOn");
      var discussionOffBtn = document.getElementById("discussionOff");
      var jsonOnBtn = document.getElementById("jsonOn");
      var jsonOffBtn = document.getElementById("jsonOff");
      var systemPromptBtn = document.getElementById("systemPrompt");
      var providerDeepseekBtn = document.getElementById("providerDeepseek");
      var providerYandexBtn = document.getElementById("providerYandex");
      var providerClaudeBtn = document.getElementById("providerClaude");
      var temperatureDeepseek = document.getElementById("temperatureDeepseek");
      var temperatureValueDeepseek = document.getElementById("temperatureValueDeepseek");
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
      var sessionId = localStorage.getItem(SESSION_KEY);
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

      function addMessage(text, role) {
        var item = document.createElement("div");
        item.className = "message " + role;
        item.textContent = text;
        messagesEl.appendChild(item);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      function setDiscussionState(isOn) {
        discussionOnBtn.classList.toggle("active", isOn);
        discussionOffBtn.classList.toggle("active", !isOn);
      }

      function setJsonState(mode) {
        jsonOnBtn.classList.toggle("active", mode === "on");
        jsonOffBtn.classList.toggle("active", mode === "off");
      }

      function setProviderState(mode) {
        providerDeepseekBtn.classList.toggle("active", mode === "deepseek");
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

      discussionOnBtn.addEventListener("click", function () {
        setDiscussionState(true);
        send("/discussion_on");
      });
      discussionOffBtn.addEventListener("click", function () {
        setDiscussionState(false);
        send("/discussion_off");
      });
      jsonOnBtn.addEventListener("click", function () {
        setJsonState("on");
        send("/json_on");
      });
      jsonOffBtn.addEventListener("click", function () {
        setJsonState("off");
        send("/json_off");
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
        setProviderState("deepseek");
        send("/use_deepseek");
      });
      providerYandexBtn.addEventListener("click", function () {
        setProviderState("yandex");
        send("/use_yandex");
      });
      providerClaudeBtn.addEventListener("click", function () {
        setProviderState("claude");
        send("/use_claude");
      });
      var temperatureRanges = {
        deepseek: { min: 0, max: 2 },
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

      function bindTemperature(inputEl, valueEl) {
        if (!inputEl || !valueEl) return;
        function syncValue() {
          valueEl.textContent = inputEl.value;
        }
        inputEl.addEventListener("input", syncValue);
        inputEl.addEventListener("change", syncValue);
        syncValue();
      }

      bindTemperature(temperatureDeepseek, temperatureValueDeepseek);
      bindTemperature(temperatureYandex, temperatureValueYandex);
      bindTemperature(temperatureClaude, temperatureValueClaude);

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
        setDiscussionState(false);
        setJsonState("on");
        setProviderState("deepseek");
        setTemperature(temperatureDeepseek, temperatureValueDeepseek, "0.5");
        setTemperature(temperatureYandex, temperatureValueYandex, "0.5");
        setTemperature(temperatureClaude, temperatureValueClaude, "0.5");
      }

      clearBtn.addEventListener("click", function () {
        requestCommand("/reset_chat")
          .then(function () {
            sessionId = generateSessionId();
            localStorage.setItem(SESSION_KEY, sessionId);
            messagesEl.innerHTML = "";
            inputEl.value = "";
            closeSystemPromptModal();
            resetUiState();
          })
          .catch(function (error) {
            addMessage("Ошибка сброса: " + error, "bot");
          });
      });

      resetUiState();
    </script>
  </body>
  </html>
    """


@app.post("/api/message")
def message(payload: MessageIn):
    session_id, user_data, chat_data = _get_session(payload.session_id)
    if payload.temperatures:
        user_data[TEMPERATURE_KEY] = normalize_temperatures(payload.temperatures)
    messages = process_text(payload.text, user_data, chat_data)
    return {"session_id": session_id, "messages": messages}

