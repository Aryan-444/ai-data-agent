"""
Visual Conversational AI Data Agent
====================================
Deployment-ready Streamlit app for Hugging Face Spaces.
Uses Google Gemini 2.5 Flash + LangChain + SQLite + Plotly Express.
"""

import os
import re
import json
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

# ──────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP — Config, Env, Page Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # picks up .env locally; HF Spaces uses st.secrets / env vars

st.set_page_config(
    page_title="AI Data Agent — Retail Insights",
    page_icon="📊",
    layout="wide",
)

# Resolve GOOGLE_API_KEY from HF Secrets OR local .env file
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = ""

# Ensure LangChain can see the key
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# SQLite database path
DB_PATH = os.path.join(os.path.dirname(__file__), "retail_store.db")

# NOTE: Actual DB columns — products(product_id, product_name, category, price)
#                         — sales(sale_id, product_id, quantity, sale_date)

# ──────────────────────────────────────────────────────────────────────────────
# 2. CACHED RESOURCES — DB Connection & LLM
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_langchain_db() -> SQLDatabase:
    """Cached LangChain SQLDatabase wrapper (schema introspection only)."""
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """Cached Gemini 2.5 Flash LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE — Safe Initialization (prevents NameError on rerun)
# ──────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

# ──────────────────────────────────────────────────────────────────────────────
# 4. AGENT TOOL — SQL Executor with Markdown Stripping
# ──────────────────────────────────────────────────────────────────────────────
_SQL_FENCE_RE = re.compile(r"```(?:sql)?(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_sql_markdown(raw: str) -> str:
    """
    Removes ```sql ... ``` or ``` ... ``` code fences the model may include
    despite being told not to. Falls back to the raw string if no fence found.
    """
    match = _SQL_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    # No fence — strip leading/trailing whitespace only
    return raw.strip()


def run_query(sql: str) -> tuple[str, pd.DataFrame | None]:
    """
    Execute *sql* against retail_store.db.
    Returns (result_string, dataframe_or_None).
    """
    clean_sql = strip_sql_markdown(sql)
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(clean_sql, conn)
        conn.close()
        result_str = df.to_string(index=False) if not df.empty else "(query returned no rows)"
        return result_str, df
    except Exception as exc:
        return f"ERROR: {exc}", None


# ──────────────────────────────────────────────────────────────────────────────
# 5. PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────
SQL_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert SQL engineer connected to a retail SQLite database.

Database Schema:
{schema}

Conversation so far (for context):
{history}

User Question: {question}

STRICT RULES:
1. Respond with ONLY the raw SQL query — no markdown fences, no explanation.
2. Use only tables and columns that exist in the schema above.
3. Prefer readable aliases (e.g., SUM(total_amount) AS total_revenue).
"""
)

RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior retail business intelligence analyst.
Synthesize a clear, concise business insight from the data below.

User Question : {question}
SQL Executed  : {query}
Query Results : {result}

PLOTTING RULE:
If the user asked for a "plot", "chart", "graph", "visualization", or "breakdown",
append a PLOT_DATA JSON block at the very end — no extra text after it.
Format (use EXACT column names from Query Results):
PLOT_DATA: {{"type": "<bar|line|pie>", "x": "<x_column>", "y": "<y_column>"}}

Do NOT include PLOT_DATA if the user did not ask for a visual.

Your Answer:"""
)

# ──────────────────────────────────────────────────────────────────────────────
# 6. CHAIN ASSEMBLY (LCEL)
# ──────────────────────────────────────────────────────────────────────────────
def build_sql_chain(db: SQLDatabase, llm: ChatGoogleGenerativeAI):
    def _get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=_get_schema)
        | SQL_PROMPT
        | llm
        | StrOutputParser()
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. PLOT PARSER — Case-Insensitive Column Matching
# ──────────────────────────────────────────────────────────────────────────────
_PLOT_TAG = "PLOT_DATA:"


def parse_and_render_plot(
    answer: str, df: pd.DataFrame | None
) -> tuple[str, object | None]:
    """
    Splits *answer* on PLOT_DATA tag, parses JSON config, returns
    (clean_text, plotly_figure_or_None).
    """
    if _PLOT_TAG not in answer:
        return answer, None

    parts = answer.split(_PLOT_TAG, maxsplit=1)
    clean_text = parts[0].strip()
    plot_json_str = parts[1].strip()

    if df is None or df.empty:
        return clean_text, None

    try:
        cfg = json.loads(plot_json_str)
        chart_type = cfg.get("type", "bar").lower()
        x_hint = cfg.get("x", "")
        y_hint = cfg.get("y", "")

        # Case-insensitive column resolution
        col_lower = {c.lower(): c for c in df.columns}
        x_col = col_lower.get(x_hint.lower(), df.columns[0])
        y_col = col_lower.get(y_hint.lower(), df.columns[-1])

        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=x_col,
                         title=f"{y_col} by {x_col}",
                         template="plotly_dark")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, markers=True,
                          title=f"{y_col} over {x_col}",
                          template="plotly_dark")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col,
                         title=f"Distribution of {y_col}",
                         template="plotly_dark")
        else:
            fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark")

        fig.update_layout(margin=dict(t=50, b=30))
        return clean_text, fig

    except Exception:
        # Never crash the app over a malformed plot config
        return clean_text, None


# ──────────────────────────────────────────────────────────────────────────────
# 8. SIDEBAR — DB Preview + Controls
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🗄️ Database Preview")
        st.caption(f"Source: `retail_store.db`")

        if not os.path.exists(DB_PATH):
            st.error("❌ `retail_store.db` not found in the repo root.")
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            for table_name, emoji in [("products", "📦"), ("sales", "📈")]:
                try:
                    df_preview = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 20", conn)
                    st.subheader(f"{emoji} {table_name.capitalize()}")
                    st.dataframe(df_preview, hide_index=True, use_container_width=True)
                except Exception:
                    st.warning(f"Could not load `{table_name}` table.")
            conn.close()
        except Exception as e:
            st.error(f"DB connection failed: {e}")

        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption("💡 Try asking:\n- *Top 5 products by revenue*\n- *Bar chart of sales by category*\n- *Which month had lowest sales?*")


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN APP LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📊 Visual Conversational AI Data Agent")
    st.markdown(
        "Ask questions in plain English. The agent generates SQL, runs it against "
        "your retail database, and renders live charts on demand."
    )

    # ── API Key Guard ──────────────────────────────────────────────────────────
    if not GOOGLE_API_KEY:
        st.error(
            "🔑 **GOOGLE_API_KEY not found.** "
            "Add it to the **Secrets** tab in your Hugging Face Space settings "
            "(or in a local `.env` file).",
            icon="🚨",
        )
        st.stop()

    # ── DB File Guard ──────────────────────────────────────────────────────────
    if not os.path.exists(DB_PATH):
        st.error(
            "❌ **`retail_store.db` not found.** "
            "Commit the database file to the root of your Hugging Face Space repository.",
            icon="🗄️",
        )
        st.stop()

    # ── Load cached resources ──────────────────────────────────────────────────
    db = get_langchain_db()
    llm = get_llm()
    sql_chain = build_sql_chain(db, llm)

    # ── Replay conversation history ────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if msg.get("fig"):
                st.plotly_chart(msg["fig"], use_container_width=True)

    # ── Chat Input ─────────────────────────────────────────────────────────────
    user_query = st.chat_input("Ask about your data or request a chart…")

    if not user_query:
        return

    # Display and store user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})

    # ── Agent Response ─────────────────────────────────────────────────────────
    with st.chat_message("assistant"):

        # Safe pre-initialisation — prevents NameError if an exception fires early
        clean_sql: str = "-- No SQL generated"
        db_result: str = ""
        df_result: pd.DataFrame | None = None
        final_answer: str = ""
        fig: object | None = None

        with st.status("🤖 Agent thinking…", expanded=True) as status:
            try:
                # Build rolling context window (last 6 turns)
                history_ctx = "\n".join(
                    f"{m['role'].capitalize()}: {m['text']}"
                    for m in st.session_state.chat_history[-6:]
                )

                # ── Step 1: Generate SQL ───────────────────────────────────────
                status.write("🔍 Step 1 — Generating SQL from your question…")
                raw_sql = sql_chain.invoke(
                    {"question": user_query, "history": history_ctx}
                )
                clean_sql = strip_sql_markdown(raw_sql)

                # ── Step 2: Execute SQL ────────────────────────────────────────
                status.write("⚙️ Step 2 — Executing query against database…")
                db_result, df_result = run_query(clean_sql)

                # ── Self-Correction Loop ───────────────────────────────────────
                if db_result.startswith("ERROR:"):
                    status.write("🔄 Query error detected — triggering repair loop…")
                    repair_question = (
                        f"The following SQL raised an error:\n```sql\n{clean_sql}\n```\n"
                        f"Error: {db_result}\n\n"
                        f"Fix the SQL so it runs correctly against the provided schema."
                    )
                    raw_sql = sql_chain.invoke(
                        {"question": repair_question, "history": history_ctx}
                    )
                    clean_sql = strip_sql_markdown(raw_sql)
                    db_result, df_result = run_query(clean_sql)

                # ── Step 3: Generate natural-language answer ───────────────────
                status.write("💬 Step 3 — Synthesising business insight…")
                response_chain = RESPONSE_PROMPT | llm | StrOutputParser()
                raw_answer = response_chain.invoke(
                    {"question": user_query, "query": clean_sql, "result": db_result}
                )

                # ── Step 4: Parse optional plot instruction ────────────────────
                final_answer, fig = parse_and_render_plot(raw_answer, df_result)

                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            except Exception as exc:
                err_str = str(exc)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    status.update(label="🛑 API Quota Exceeded", state="error")
                    final_answer = (
                        "⚠️ **Gemini API quota limit reached (HTTP 429).**\n\n"
                        "The free tier allows ~20 requests/day. Options:\n"
                        "- Wait until quota resets (midnight Pacific)\n"
                        "- Upgrade to a paid API key\n"
                        "- Add a new `GOOGLE_API_KEY` in your Space Secrets"
                    )
                else:
                    status.update(label="❌ Unexpected Error", state="error")
                    final_answer = (
                        f"An unexpected error occurred:\n\n```\n{err_str}\n```"
                    )

        # ── Render outputs ─────────────────────────────────────────────────────
        st.write(final_answer)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("🛠️ Execution Trace", expanded=False):
            st.markdown("**Generated SQL:**")
            st.code(clean_sql, language="sql")
            st.markdown("**Raw Query Result:**")
            if df_result is not None and not df_result.empty:
                st.dataframe(df_result, hide_index=True, use_container_width=True)
            else:
                st.text(db_result or "(no data)")

        # ── Persist assistant turn to session state ────────────────────────────
        entry: dict = {"role": "assistant", "text": final_answer}
        if fig is not None:
            entry["fig"] = fig
        st.session_state.chat_history.append(entry)

# Streamlit executes this file as a module (not via __main__),
# so main() MUST be called unconditionally at the top level.
main()
