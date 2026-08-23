"""
Concept Explainer — Multi-Provider LLM Chat App
=================================================
A Streamlit app that lets you plug in an OpenAI or Anthropic API key,
validates the key with a real API call, lists every model available to
that key alongside best-effort pricing, and then gives you a chat panel
that explains any concept (or set of concepts) to every kind of audience
at once — plain-English summary, analogy, technical depth, SME/domain
relevance, and business impact — with follow-up Q&A and a text export
of the whole conversation.

Deploy on Streamlit Community Cloud:
1. Push this file + requirements.txt to a GitHub repo.
2. On share.streamlit.io, point a new app at app.py.
3. No secrets needed — users paste their own API key in the sidebar.
"""

import json
from datetime import datetime

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Concept Explainer",
    page_icon="💡",
    layout="wide",
)

# --------------------------------------------------------------------------
# Best-effort pricing tables (USD per 1M tokens, standard tier).
# Prices change often — these are approximate as of Aug 2026 and are only
# a convenience display. Always confirm against the provider's official
# pricing page before making cost decisions. Matching is by prefix so new
# dated snapshots of a model (e.g. "gpt-5.6-sol-2026-07-09") still resolve.
# --------------------------------------------------------------------------
OPENAI_PRICING = {
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (1.25, 7.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o1": (15.00, 60.00),
}

ANTHROPIC_PRICING = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def lookup_price(model_id: str, table: dict):
    """Prefix-match a model id against a pricing table."""
    mid = model_id.lower()
    best = None
    for prefix, price in table.items():
        if mid.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, price)
    return best[1] if best else None


# --------------------------------------------------------------------------
# Provider API helpers
# --------------------------------------------------------------------------
def validate_and_list_openai(api_key: str):
    """Calls OpenAI's /v1/models — doubles as both auth check and model list."""
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI key check failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json().get("data", [])
    # Keep it to chat-relevant model families, drop embeddings/tts/whisper/etc.
    keep_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
    drop_contains = ("audio", "embedding", "whisper", "tts", "moderation", "image", "realtime", "search", "transcribe")
    models = []
    for m in data:
        mid = m.get("id", "")
        if not mid.startswith(keep_prefixes):
            continue
        if any(d in mid for d in drop_contains):
            continue
        price = lookup_price(mid, OPENAI_PRICING)
        models.append({"id": mid, "input_price": price[0] if price else None,
                        "output_price": price[1] if price else None})
    models.sort(key=lambda m: m["id"])
    return models


def validate_and_list_anthropic(api_key: str):
    """Calls Anthropic's /v1/models — doubles as both auth check and model list."""
    resp = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic key check failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json().get("data", [])
    models = []
    for m in data:
        mid = m.get("id", "")
        price = lookup_price(mid, ANTHROPIC_PRICING)
        models.append({"id": mid, "input_price": price[0] if price else None,
                        "output_price": price[1] if price else None})
    models.sort(key=lambda m: m["id"])
    return models


def call_openai_chat(api_key: str, model: str, messages: list) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.6},
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI request failed ({resp.status_code}): {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def call_anthropic_chat(api_key: str, model: str, system: str, messages: list) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2000,
            "system": system,
            "messages": messages,
        },
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic request failed ({resp.status_code}): {resp.text[:400]}")
    content_blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")


# --------------------------------------------------------------------------
# The system prompt that shapes every explanation
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert technical educator and communicator who specializes in \
explaining technology and business concepts to mixed audiences in a single room: engineers, \
subject-matter experts (SMEs) with no coding background, domain/business experts, and executives.

When the user gives you a concept or set of concepts (e.g. "NLP", "vector databases + RAG"), \
produce a response structured like this, using Markdown headers:

### In Plain English
1-3 sentences, zero jargon, that anyone could understand immediately.

### Analogy
One vivid, relatable analogy (everyday life, not another technical field) that makes the \
mechanism of the concept click intuitively.

### How It Actually Works (Technical Depth)
A precise, technically accurate explanation for engineers/practitioners — key mechanisms, \
important terminology, common pitfalls or tradeoffs. Do not water this down; this is for people \
who will implement or evaluate it.

### Why It Matters to a Domain Expert / SME
Translate the concept into the language of someone who owns a business process or domain \
(e.g. finance, healthcare, operations, legal) but isn't technical — what changes for them, what \
new capability or risk this introduces, how they'd recognize it in a vendor pitch or a project plan.

### Business Impact
Concrete commercial relevance: cost, efficiency, revenue, risk/compliance, competitive positioning, \
and realistic ROI considerations. Be specific rather than generic ("saves time") where possible.

### Key Takeaways
3-5 crisp bullet points summarizing the above.

Rules:
- Always keep technical accuracy — never sacrifice correctness for simplicity, layer the depth instead.
- If multiple concepts are given, briefly explain each, then add a short section on how they relate.
- For follow-up questions, answer directly and conversationally; you don't have to repeat the full \
structure above unless it genuinely helps, but keep the same "no jargon left unexplained, analogy \
when useful, tie back to business impact" spirit.
- Keep formatting clean and skimmable. Avoid filler.
"""


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
defaults = {
    "provider": "OpenAI",
    "api_key": "",
    "validated": False,
    "models": [],
    "selected_model": None,
    "messages": [],  # list of {"role": "user"/"assistant", "content": str}
    "validation_error": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------
# Sidebar — provider setup
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("💡 Setup")

    provider = st.radio("Provider", ["OpenAI", "Anthropic"], horizontal=True,
                         index=0 if st.session_state.provider == "OpenAI" else 1)
    if provider != st.session_state.provider:
        # Reset validation state on provider switch
        st.session_state.provider = provider
        st.session_state.validated = False
        st.session_state.models = []
        st.session_state.selected_model = None

    api_key = st.text_input(f"{provider} API key", type="password",
                             value=st.session_state.api_key,
                             help="Your key is only used in this session and is never stored.")
    st.session_state.api_key = api_key

    validate_clicked = st.button("🔐 Validate key & load models", use_container_width=True)

    if validate_clicked:
        if not api_key.strip():
            st.session_state.validated = False
            st.session_state.validation_error = "Please enter an API key first."
        else:
            with st.spinner(f"Calling {provider} to validate the key and fetch models…"):
                try:
                    if provider == "OpenAI":
                        models = validate_and_list_openai(api_key.strip())
                    else:
                        models = validate_and_list_anthropic(api_key.strip())
                    st.session_state.models = models
                    st.session_state.validated = True
                    st.session_state.validation_error = None
                    if models:
                        st.session_state.selected_model = models[0]["id"]
                except Exception as e:
                    st.session_state.validated = False
                    st.session_state.validation_error = str(e)

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    if st.session_state.validated and st.session_state.models:
        st.success(f"Key validated — {len(st.session_state.models)} models available.")

        st.markdown("**Available models & pricing** (USD / 1M tokens)")
        table_rows = []
        for m in st.session_state.models:
            ip = f"${m['input_price']:.2f}" if m["input_price"] is not None else "—"
            op = f"${m['output_price']:.2f}" if m["output_price"] is not None else "—"
            table_rows.append({"Model": m["id"], "Input": ip, "Output": op})
        st.dataframe(table_rows, use_container_width=True, hide_index=True, height=260)
        st.caption("Prices are best-effort static estimates and may be stale — verify on the "
                   "provider's official pricing page before budgeting.")

        model_ids = [m["id"] for m in st.session_state.models]
        default_idx = model_ids.index(st.session_state.selected_model) if st.session_state.selected_model in model_ids else 0
        st.session_state.selected_model = st.selectbox("Model to chat with", model_ids, index=default_idx)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript_lines = [f"Concept Explainer conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                             f"Provider: {st.session_state.provider}  |  Model: {st.session_state.selected_model}",
                             "=" * 60, ""]
        for m in st.session_state.messages:
            speaker = "YOU" if m["role"] == "user" else "ASSISTANT"
            transcript_lines.append(f"[{speaker}]\n{m['content']}\n")
        transcript = "\n".join(transcript_lines)
        st.download_button("⬇️ Download conversation (.txt)", data=transcript,
                            file_name=f"concept_explainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain", use_container_width=True)


# --------------------------------------------------------------------------
# Main panel — chat
# --------------------------------------------------------------------------
st.title("Concept Explainer")
st.caption("Enter a concept or a set of concepts (e.g. \"NLP\", \"RAG + vector databases\") and get "
           "an explanation layered for everyone in the room — plain English, an analogy, technical "
           "depth, SME relevance, and business impact. Ask follow-ups any time.")

if not st.session_state.validated:
    st.info("👈 Enter and validate an API key in the sidebar to start chatting.")
else:
    # Render existing history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    placeholder = "e.g. Explain 'zero-knowledge proofs' or 'RAG + fine-tuning'…" if not st.session_state.messages \
        else "Ask a follow-up question…"
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking through the explanation…"):
                try:
                    if st.session_state.provider == "OpenAI":
                        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                        reply = call_openai_chat(st.session_state.api_key, st.session_state.selected_model, api_messages)
                    else:
                        api_messages = [{"role": r["role"], "content": r["content"]} for r in st.session_state.messages]
                        reply = call_anthropic_chat(st.session_state.api_key, st.session_state.selected_model,
                                                     SYSTEM_PROMPT, api_messages)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    err = f"⚠️ Request failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
