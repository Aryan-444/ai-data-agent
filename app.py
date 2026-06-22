"""
Visual Conversational AI Data Agent
====================================
Deployment-ready Streamlit app for Hugging Face Spaces.
Uses Google Gemini 2.5 Flash + LangChain + SQLite + Plotly Express.
Supports dynamic dataset upload: SQLite .db files OR CSV files.
"""

import os
import re
import json
import uuid
import sqlite3
import tempfile

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
load_dotenv()

st.set_page_config(
    page_title="AI Data Agent — Universal SQL Assistant",
    page_icon="📊",
    layout="wide",
)

# Resolve GOOGLE_API_KEY from HF Secrets OR local .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = ""

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Default bundled database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "retail_store.db")

# ──────────────────────────────────────────────────────────────────────────────
# 2. SESSION STATE — Safe Initialization
# ──────────────────────────────────────────────────────────────────────────────
if "chat_history"      not in st.session_state:
    st.session_state.chat_history      = []
if "active_db_path"    not in st.session_state:
    st.session_state.active_db_path    = DEFAULT_DB_PATH
if "active_db_label"   not in st.session_state:
    st.session_state.active_db_label   = "retail_store.db (default)"
if "session_id"        not in st.session_state:
    # Unique ID per browser session — keeps uploaded files isolated
    st.session_state.session_id        = str(uuid.uuid4())[:8]
if "uploaded_tables"   not in st.session_state:
    st.session_state.uploaded_tables   = []

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATASET UPLOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_session_tmp_dir() -> str:
    """Returns a session-scoped temp directory (created once per session)."""
    tmp_base = tempfile.gettempdir()
    session_dir = os.path.join(tmp_base, f"ai_agent_{st.session_state.session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_uploaded_sqlite(uploaded_file) -> str:
    """Save an uploaded .db file to the session temp dir. Returns the path."""
    session_dir = get_session_tmp_dir()
    db_path = os.path.join(session_dir, uploaded_file.name)
    with open(db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return db_path


def csvs_to_sqlite(uploaded_files) -> tuple[str, list[str]]:
    """
    Convert one or more uploaded CSV files into a single SQLite database.
    Each CSV becomes one table (named after the file, without extension).
    Returns (db_path, list_of_table_names).
    """
    session_dir = get_session_tmp_dir()
    db_path = os.path.join(session_dir, "user_upload.db")

    # Remove old db if rebuilding
    if os.path.exists(db_path):
        os.remove(db_path)

    table_names = []
    conn = sqlite3.connect(db_path)

    for f in uploaded_files:
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(f.name)[0])
        try:
            df = pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")

        # Sanitize column names
        df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c).strip("_") for c in df.columns]
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        table_names.append(table_name)

    conn.close()
    return db_path, table_names


def get_table_names(db_path: str) -> list[str]:
    """Return list of user table names in a SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# 4. CACHED RESOURCES — DB Connection (keyed on path) & LLM
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_langchain_db(db_path: str) -> SQLDatabase:
    """Cached LangChain SQLDatabase — one instance per unique db_path."""
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """Cached Gemini 2.5 Flash LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. SQL UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
_SQL_FENCE_RE = re.compile(r"```(?:sql)?(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_sql_markdown(raw: str) -> str:
    match = _SQL_FENCE_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def run_query(sql: str, db_path: str) -> tuple[str, pd.DataFrame | None]:
    """Execute SQL against the given db_path, return (result_str, df)."""
    clean_sql = strip_sql_markdown(sql)
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(clean_sql, conn)
        conn.close()
        result_str = df.to_string(index=False) if not df.empty else "(query returned no rows)"
        return result_str, df
    except Exception as exc:
        return f"ERROR: {exc}", None


# ──────────────────────────────────────────────────────────────────────────────
# 6. PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────
SQL_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert SQL engineer. The user has connected a SQLite database.

Database Schema:
{schema}

Conversation so far:
{history}

User Question: {question}

STRICT RULES:
1. Respond with ONLY the raw SQL query — no markdown fences, no explanation.
2. Use ONLY tables and columns that exist in the schema above.
3. Use readable aliases (e.g., SUM(amount) AS total_revenue).
"""
)

RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior data analyst. Synthesize a clear insight from the data below.

User Question : {question}
SQL Executed  : {query}
Query Results : {result}

PLOTTING RULE:
If the user asked for a "plot", "chart", "graph", "visualization", or "breakdown",
append a PLOT_DATA JSON block at the very end — no extra text after it.
Format (use EXACT column names from Query Results):
PLOT_DATA: {{"type": "<bar|line|pie>", "x": "<x_column>", "y": "<y_column>"}}

Do NOT include PLOT_DATA if no visual was requested.

Your Answer:"""
)


# ──────────────────────────────────────────────────────────────────────────────
# 7. CHAIN ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────
def build_sql_chain(db: SQLDatabase, llm):
    def _get_schema(_):
        return db.get_table_info()
    return (
        RunnablePassthrough.assign(schema=_get_schema)
        | SQL_PROMPT
        | llm
        | StrOutputParser()
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. PLOT PARSER
# ──────────────────────────────────────────────────────────────────────────────
_PLOT_TAG = "PLOT_DATA:"


def parse_and_render_plot(answer: str, df: pd.DataFrame | None) -> tuple[str, object | None]:
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
        col_lower = {c.lower(): c for c in df.columns}
        x_col = col_lower.get(cfg.get("x", "").lower(), df.columns[0])
        y_col = col_lower.get(cfg.get("y", "").lower(), df.columns[-1])

        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=x_col,
                         title=f"{y_col} by {x_col}", template="plotly_dark")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, markers=True,
                          title=f"{y_col} over {x_col}", template="plotly_dark")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col,
                         title=f"Distribution of {y_col}", template="plotly_dark")
        else:
            fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark")

        fig.update_layout(margin=dict(t=50, b=30))
        return clean_text, fig
    except Exception:
        return clean_text, None


# ──────────────────────────────────────────────────────────────────────────────
# 9. SIDEBAR — Dataset Upload + DB Preview + Controls
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("📊 AI Data Agent")

        # ── Dataset Upload Section ────────────────────────────────────────────
        st.header("📤 Upload Your Dataset")
        st.caption("Replace the default database with your own data.")

        data_source = st.radio(
            "Data source",
            ["🏪 Default (retail_store.db)", "🗄️ Upload SQLite .db file", "📄 Upload CSV file(s)"],
            key="data_source_radio",
        )

        if data_source == "🗄️ Upload SQLite .db file":
            uploaded_db = st.file_uploader(
                "Upload a .db / .sqlite file",
                type=["db", "sqlite", "sqlite3"],
                key="sqlite_uploader",
                help="Your entire SQLite database — all tables will be available.",
            )
            if uploaded_db is not None:
                with st.spinner("Loading database…"):
                    try:
                        new_path = save_uploaded_sqlite(uploaded_db)
                        tables   = get_table_names(new_path)
                        if not tables:
                            st.error("No tables found in the uploaded database.")
                        elif new_path != st.session_state.active_db_path:
                            st.session_state.active_db_path  = new_path
                            st.session_state.active_db_label = uploaded_db.name
                            st.session_state.uploaded_tables = tables
                            st.session_state.chat_history    = []
                            st.success(f"Loaded! Tables: {', '.join(tables)}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

        elif data_source == "📄 Upload CSV file(s)":
            uploaded_csvs = st.file_uploader(
                "Upload one or more CSV files",
                type=["csv"],
                accept_multiple_files=True,
                key="csv_uploader",
                help="Each CSV becomes one table. File name = table name.",
            )
            if uploaded_csvs:
                if st.button("▶ Build Database from CSVs", use_container_width=True):
                    with st.spinner("Converting CSVs to SQLite…"):
                        try:
                            new_path, tables = csvs_to_sqlite(uploaded_csvs)
                            st.session_state.active_db_path  = new_path
                            st.session_state.active_db_label = f"{len(tables)} CSV table(s)"
                            st.session_state.uploaded_tables = tables
                            st.session_state.chat_history    = []
                            st.success(f"Ready! Tables: {', '.join(tables)}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Conversion failed: {e}")

        else:  # Default
            if st.session_state.active_db_path != DEFAULT_DB_PATH:
                if st.button("↩ Restore Default Database", use_container_width=True):
                    st.session_state.active_db_path  = DEFAULT_DB_PATH
                    st.session_state.active_db_label = "retail_store.db (default)"
                    st.session_state.uploaded_tables = []
                    st.session_state.chat_history    = []
                    st.rerun()

        # Active DB indicator
        st.info(f"**Active DB:** `{st.session_state.active_db_label}`", icon="🗄️")
        st.divider()

        # ── Table Preview ─────────────────────────────────────────────────────
        st.header("🗄️ Table Preview")
        active_path = st.session_state.active_db_path

        if not os.path.exists(active_path):
            st.error("Database file not found.")
        else:
            tables = get_table_names(active_path)
            if not tables:
                st.warning("No tables found.")
            else:
                try:
                    conn = sqlite3.connect(active_path)
                    for tbl in tables:
                        with st.expander(f"📋 {tbl}", expanded=(len(tables) == 1)):
                            try:
                                df_prev = pd.read_sql_query(
                                    f"SELECT * FROM [{tbl}] LIMIT 10", conn
                                )
                                st.caption(f"{len(df_prev)} rows shown (max 10) · {len(df_prev.columns)} columns")
                                st.dataframe(df_prev, hide_index=True, use_container_width=True)
                            except Exception as te:
                                st.warning(f"Could not preview: {te}")
                    conn.close()
                except Exception as e:
                    st.error(f"DB error: {e}")

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────────
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption(
            "💡 **Example queries:**\n"
            "- *Show top 5 rows*\n"
            "- *How many records in total?*\n"
            "- *Bar chart of sales by category*\n"
            "- *Which month had highest revenue?*"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 10. MAIN APP LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📊 Visual Conversational AI Data Agent")
    st.markdown(
        "Ask questions in plain English about **any SQLite database or CSV dataset**. "
        "Upload your own data using the sidebar, or use the default retail store database."
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
    active_db = st.session_state.active_db_path
    if not os.path.exists(active_db):
        st.error(
            "❌ **Database not found.** "
            "Upload a database using the sidebar, or commit `retail_store.db` to the repo.",
            icon="🗄️",
        )
        st.stop()

    # ── Load cached resources ──────────────────────────────────────────────────
    # get_langchain_db is keyed on db_path — switching DB auto-creates new connection
    db        = get_langchain_db(active_db)
    llm       = get_llm()
    sql_chain = build_sql_chain(db, llm)

    # ── Replay conversation history ────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if msg.get("fig"):
                st.plotly_chart(msg["fig"], use_container_width=True)

    # ── Chat Input ─────────────────────────────────────────────────────────────
    user_query = st.chat_input(
        f"Ask about your data [{st.session_state.active_db_label}]…"
    )

    if not user_query:
        return

    # Display and store user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})

    # ── Agent Response ─────────────────────────────────────────────────────────
    with st.chat_message("assistant"):

        # Safe pre-initialisation
        clean_sql    : str                = "-- No SQL generated"
        db_result    : str                = ""
        df_result    : pd.DataFrame | None = None
        final_answer : str                = ""
        fig          : object | None      = None

        with st.status("🤖 Agent thinking…", expanded=True) as status:
            try:
                history_ctx = "\n".join(
                    f"{m['role'].capitalize()}: {m['text']}"
                    for m in st.session_state.chat_history[-6:]
                )

                # Step 1 — Generate SQL
                status.write("🔍 Step 1 — Generating SQL from your question…")
                raw_sql   = sql_chain.invoke({"question": user_query, "history": history_ctx})
                clean_sql = strip_sql_markdown(raw_sql)

                # Step 2 — Execute SQL
                status.write("⚙️ Step 2 — Executing query against database…")
                db_result, df_result = run_query(clean_sql, active_db)

                # Self-Correction Loop
                if db_result.startswith("ERROR:"):
                    status.write("🔄 Query error — triggering repair loop…")
                    repair_q = (
                        f"The following SQL raised an error:\n```sql\n{clean_sql}\n```\n"
                        f"Error: {db_result}\n\nFix the SQL to run correctly against the schema."
                    )
                    raw_sql   = sql_chain.invoke({"question": repair_q, "history": history_ctx})
                    clean_sql = strip_sql_markdown(raw_sql)
                    db_result, df_result = run_query(clean_sql, active_db)

                # Step 3 — Synthesise answer
                status.write("💬 Step 3 — Synthesising business insight…")
                response_chain = RESPONSE_PROMPT | llm | StrOutputParser()
                raw_answer = response_chain.invoke(
                    {"question": user_query, "query": clean_sql, "result": db_result}
                )

                # Step 4 — Parse optional plot
                final_answer, fig = parse_and_render_plot(raw_answer, df_result)

                status.update(label="Analysis complete!", state="complete", expanded=False)

            except Exception as exc:
                err_str = str(exc)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    status.update(label="API Quota Exceeded", state="error")
                    final_answer = (
                        "**Gemini API quota limit reached (HTTP 429).**\n\n"
                        "- Wait until quota resets (midnight Pacific)\n"
                        "- Or add a new `GOOGLE_API_KEY` in Space Secrets"
                    )
                else:
                    status.update(label="Unexpected Error", state="error")
                    final_answer = f"An error occurred:\n\n```\n{err_str}\n```"

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

        # Persist to session state
        entry: dict = {"role": "assistant", "text": final_answer}
        if fig is not None:
            entry["fig"] = fig
        st.session_state.chat_history.append(entry)


# Streamlit executes this file as a module (not via __main__),
# so main() MUST be called unconditionally at the top level.
main()
