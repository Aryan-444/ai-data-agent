---
title: Visual Conversational AI Data Agent
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
license: mit
short_description: Text-to-SQL AI agent with live charts
---

# 📊 Visual Conversational AI Data Agent

A production-grade **Text-to-SQL AI Agent** built with **Google Gemini 2.5 Flash**, LangChain, and Streamlit. Ask questions in plain English and get instant SQL-powered insights with live Plotly visualizations.

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **LLM Backend** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) via `langchain-google-genai` |
| 🗄️ **Database** | SQLite (`retail_store.db`) — Products & Sales tables |
| 💬 **Memory** | Multi-turn `st.session_state` chat history (last 6 turns in context) |
| 🔁 **Self-Correction** | Automatic SQL repair loop on query failure |
| 📈 **Visualizations** | Plotly Express — bar, line, and pie charts on demand |
| 🛡️ **Quota Guard** | Graceful 429 / `RESOURCE_EXHAUSTED` error handling |

## 🚀 How to Use

1. Set your `GOOGLE_API_KEY` in the **Secrets** tab of your Hugging Face Space.
2. Ask natural language questions like:
   - *"What are the top 5 products by total revenue?"*
   - *"Show me a bar chart of sales quantity per category"*
   - *"Which month had the highest sales?"*

## 🔑 Environment Variable Required

```
GOOGLE_API_KEY = your_google_ai_studio_key_here
```

> Get your free key at [aistudio.google.com](https://aistudio.google.com)

## 🗂️ Repository Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── retail_store.db      # SQLite database (committed to repo)
└── README.md            # This file
```

## 📦 Database Schema

The `retail_store.db` includes:
- **products** — `product_id`, `product_name`, `category`, `price`
- **sales** — `sale_id`, `product_id`, `quantity`, `sale_date`
