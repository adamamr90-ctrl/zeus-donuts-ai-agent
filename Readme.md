# Zeus Donuts — AI Customer Service Agent 

An AI-powered customer service chatbot for a donut shop, built with
**LangGraph**, **LangChain**, and **Groq**. It handles product
questions, live inventory checks, multi-item orders, order
modifications/cancellations, and FAQ questions — all in **Egyptian
Arabic** — while keeping a real Google Sheet as the single source of
truth for stock.

---

## Why this project exists

Most "AI agent" demos manage a fixed, static inventory. This one
doesn't: the shop sells donuts both **online (chat)** and **in-person
(walk-in customers)**, so the AI's inventory has to reflect a stock
level that can change from *outside* the bot at any moment. Google
Sheets was chosen specifically so the shop owner can update stock
from their phone in seconds, with the agent always reading the live
numbers.

---

## Architecture

```
User (Streamlit chat)
        │
        ▼
   app.py  ── owns the system prompt (SOP) + per-session state
        │
        ▼
  agent.py ── LangGraph state machine
        │
   ┌────┴─────┐
   │  agent   │  <-- calls the LLM, decides: reply or call a tool?
   │  node    │
   └────┬─────┘
        │ tool call?
        ▼
   ┌──────────┐
   │  tools   │  <-- executes the requested tool(s)
   │  node    │
   └────┬─────┘
        │
        ▼ (loops back to agent until no more tool calls)
      tools.py
        │
   ┌────┴────────────────┬─────────────────┐
   ▼                      ▼                 ▼
Google Sheets       Chroma (RAG)      Pydantic validation
(live inventory)    (Arabic FAQ)      (order item schema)
```

**Key design decisions:**

- **Two-pass order validation** — every order is checked for
  availability *before* any inventory is touched. If any single item
  in a multi-item order can't be fulfilled, nothing is deducted. This
  prevents partial, inconsistent stock updates.
- **Structured tool outputs** — every tool returns a JSON-shaped
  `{"success", "data", "message"}` dict instead of a raw string. This
  gives the LLM unambiguous data to work with and noticeably reduced
  hallucinated responses compared to free-text tool outputs.
- **Explicit SOP-style system prompt** — the agent's behavior is
  defined as a numbered procedure (intent classification → tool
  selection → clarification rules → output rules) rather than a vague
  paragraph, which made its behavior far more consistent.
- **Per-user conversation memory** — each browser session gets its
  own `thread_id`, so LangGraph's checkpointer keeps every customer's
  conversation completely isolated.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM orchestration | LangGraph, LangChain |
| LLM provider | Groq (`llama-3.3-70b-versatile`) |
| Live inventory | Google Sheets API (`gspread`) |
| Knowledge base | Chroma + Sentence-Transformer embeddings (RAG) |
| Data validation | Pydantic |
| UI | Streamlit |

---

## Features

- 🍩 Real-time inventory lookup (single flavor or full menu)
- 🛒 Multi-item order placement with atomic validation
- ✏️ Order modification (add / remove / change quantity)
- ❌ Order cancellation with automatic stock restoration
- 📖 RAG-powered FAQ answering (hours, delivery, payment, flavor
  descriptions) in Arabic
- 🗣️ Clarifying questions instead of guessing on ambiguous requests
  (e.g. an unclear flavor name, missing quantity)

---

## Project Structure

```
zeus_donuts_bot/
├── app.py                 # Streamlit UI + system prompt (SOP)
├── agent.py                # LangGraph graph (agent/tools nodes)
├── tools.py                 # All AI-callable tools
├── sheets_connection.py     # Google Sheets client setup
├── rag_setup.py              # Chroma vector store setup for FAQ
├── faq.txt                    # Arabic FAQ knowledge base source
├── credentials.json            # Google service account key (gitignored)
└── .env                          # API keys (gitignored)
```

---

## Setup

```bash
pip install streamlit langgraph langchain-groq langchain-core \
    gspread google-auth langchain-chroma \
    langchain-community sentence-transformers python-dotenv
```

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Set up a Google Cloud service account, enable the Sheets + Drive
APIs, download the credentials JSON as `credentials.json`, and share
your inventory spreadsheet with the service account's email.

Run:

```bash
streamlit run app.py
```

---

## What I'd improve next

- Separate a customer's **cart** (in-progress, unconfirmed order) from
  the **confirmed order** that actually touches the Google Sheet, so
  the LLM never has to "remember and recalculate" a multi-step order
  on its own.
- Swap `InMemorySaver` for a persistent, database-backed checkpointer
  for real production use.
- Connect to the Instagram/Messenger Graph API via a webhook so the
  same agent can serve customers directly in DMs.