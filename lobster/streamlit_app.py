# streamlit_app.py
"""
Lobster AI - Streamlit Interface (stable, one-shot chat execution)
- Immediate message display
- Single invocation per prompt
- Clean logs & plot rendering
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from rich.console import Console

# --- Your core stack ---
from lobster.core.client import AgentClient
from lobster.core.data_manager import DataManager
from lobster.utils.callbacks import TerminalCallbackHandler
from lobster.config.agent_config import get_agent_configurator, initialize_configurator

# -----------------------
# Basic setup & styling
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🦞 Lobster AI - Bioinformatics Analysis",
    page_icon="🦞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .terminal-output { background:#1e1e1e;color:#00ff00;font-family:'Courier New',monospace;padding:10px;border-radius:6px;max-height:420px;overflow-y:auto;white-space:pre-wrap;font-size:12px; }
    .status-box { background:#f8f9fa;padding:8px;border-radius:6px;margin:5px 0;font-size:13px;border-left:4px solid #ff4b4b; }
    .status-item { display:flex;justify-content:space-between;align-items:center;margin:3px 0;font-size:12px; }
    .status-label { font-weight:600;color:#666; }
    .status-value { color:#333; }
    .plot-container { margin: 10px 0; padding: 10px; background:#f8f9fa; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# -----------------------
# Logging helpers
# -----------------------
class SimpleLogCollector:
    def __init__(self):
        self.logs = []
        self.step = 0

    def add_log(self, level: str, agent: str, message: str):
        self.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "agent": agent,
            "message": message,
            "step": self.step
        })
        self.step += 1

    def get_formatted_logs(self) -> str:
        if not self.logs:
            return "No logs available"
        icon_map = {"START":"🤖","TOOL":"🔧","COMPLETE":"✅","ERROR":"❌","INFO":"ℹ️"}
        out = []
        for log in self.logs:
            agent = (log["agent"] or "System").replace("_"," ").title()
            icon = icon_map.get(log["level"], "•")
            out.append(f"{log['timestamp']} {icon} {agent}: {log['message']}")
        return "\n".join(out)

    def clear(self):
        self.logs.clear()
        self.step = 0


class StreamlitCallbackHandler(TerminalCallbackHandler):
    """Collect logs safely for Streamlit without threading issues."""
    def __init__(self, log_collector: SimpleLogCollector = None, *args, **kwargs):
        dummy_console = Console(file=StringIO(), force_terminal=False)
        super().__init__(console=dummy_console, *args, **kwargs)
        self.log_collector = log_collector or SimpleLogCollector()
        self.current_agent = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        agent_name = kwargs.get("name") or (serialized.get("name", "unknown") if serialized else "unknown")
        self.current_agent = agent_name
        self.log_collector.add_log("START", agent_name, "Starting analysis...")

    def on_llm_end(self, response, **kwargs) -> None:
        if self.current_agent:
            self.log_collector.add_log("COMPLETE", self.current_agent, "Analysis complete")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "unknown_tool") if serialized else "unknown_tool"
        agent = self.current_agent or "system"
        self.log_collector.add_log("TOOL", agent, f"Using tool: {tool_name}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        agent = self.current_agent or "system"
        self.log_collector.add_log("COMPLETE", agent, "Tool execution complete")

    def on_tool_error(self, error, **kwargs) -> None:
        agent = self.current_agent or "system"
        self.log_collector.add_log("ERROR", agent, f"Tool error: {str(error)[:200]}...")


# -----------------------
# Session init
# -----------------------
def init_session_state():
    if "initialized" in st.session_state:
        return

    # Workspace
    workspace_path = Path.cwd() / ".lobster_workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    st.session_state.initialized = True
    st.session_state.workspace_path = workspace_path
    st.session_state.data_manager = DataManager(workspace_path=workspace_path)

    # Chat / agent state
    st.session_state.messages: List[Dict[str, Any]] = []
    st.session_state.client: Optional[AgentClient] = None
    st.session_state.current_mode = "production"
    st.session_state.enable_reasoning = True
    st.session_state.show_terminal = True
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.processing = False  # lock to prevent duplicate runs
    st.session_state.saved_uploads = {}  # name -> size fingerprint

    # Logging callback
    st.session_state.log_collector = SimpleLogCollector()
    st.session_state.callbacks = [
        StreamlitCallbackHandler(
            log_collector=st.session_state.log_collector,
            show_reasoning=st.session_state.enable_reasoning,
            verbose=True
        )
    ]

    # Ensure a default configurator profile exists
    try:
        initialize_configurator(profile=st.session_state.current_mode)
    except Exception as e:
        logger.warning(f"Configurator init warning: {e}")


def init_client() -> AgentClient:
    if st.session_state.client is None:
        st.session_state.client = AgentClient(
            data_manager=st.session_state.data_manager,
            session_id=st.session_state.session_id,
            workspace_path=st.session_state.workspace_path,
            enable_reasoning=st.session_state.enable_reasoning,
            custom_callbacks=st.session_state.callbacks
        )
    return st.session_state.client


# -----------------------
# Sidebar
# -----------------------
def display_sidebar():
    dm: DataManager = st.session_state.data_manager
    st.sidebar.markdown("## 🦞 **Lobster AI**")
    st.sidebar.markdown("*Multi-Agent Bioinformatics System*")
    st.sidebar.markdown("---")

    with st.sidebar.expander("📊 **System Status**", expanded=True):
        st.markdown(
            f"""
            <div class="status-box">
                <div class="status-item"><span class="status-label">Session:</span><span class="status-value">{st.session_state.session_id[:10]}…</span></div>
                <div class="status-item"><span class="status-label">Mode:</span><span class="status-value">{'🟢' if st.session_state.current_mode=='production' else '🔵'} {st.session_state.current_mode}</span></div>
            """,
            unsafe_allow_html=True,
        )
        if dm.has_data():
            summary = dm.get_data_summary()
            st.markdown(
                f"""<div class="status-item"><span class="status-label">Data:</span><span class="status-value">✓ {summary['shape'][0]} × {summary['shape'][1]}</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="status-item"><span class="status-label">Data:</span><span class="status-value">No data loaded</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="status-item"><span class="status-label">Plots:</span><span class="status-value">📈 {len(dm.latest_plots)}</span></div></div>""",
            unsafe_allow_html=True,
        )

    with st.sidebar.expander("⚙️ **Configuration**", expanded=False):
        configurator = get_agent_configurator()
        profiles = list(getattr(configurator, "list_available_profiles")().keys())
        # Fallback if production not present
        if st.session_state.current_mode not in profiles and profiles:
            st.session_state.current_mode = profiles[0]

        selected = st.selectbox("Operation Mode", options=profiles, index=profiles.index(st.session_state.current_mode))
        if selected != st.session_state.current_mode and st.button("🔄 Apply Mode"):
            st.session_state.current_mode = selected
            try:
                initialize_configurator(profile=selected)
            except Exception as e:
                st.sidebar.error(f"Configurator error: {e}")
            # Recreate client next run
            st.session_state.client = None

        st.session_state.enable_reasoning = st.checkbox("Show Agent Reasoning", value=st.session_state.enable_reasoning)
        st.session_state.show_terminal = st.checkbox("Show Terminal Output", value=st.session_state.show_terminal)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 **Upload File**")
    uploaded = st.sidebar.file_uploader("Choose a file", type=['csv', 'tsv', 'xlsx', 'xls', 'txt', 'json', 'h5', 'h5ad'])
    if uploaded is not None:
        # De-duplicate saves across reruns
        size = getattr(uploaded, "size", None)
        key = f"{uploaded.name}:{size}"
        if st.session_state.saved_uploads.get(uploaded.name) != size:
            data_dir = st.session_state.workspace_path / "data"
            data_dir.mkdir(exist_ok=True)
            file_path = data_dir / uploaded.name
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.saved_uploads[uploaded.name] = size
            st.sidebar.success(f"✓ Uploaded: {uploaded.name}")

            # Try load basic tabular
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(file_path, index_col=0)
                elif uploaded.name.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t", index_col=0)
                elif uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path, index_col=0)
                else:
                    df = None  # handled by agent later
                if df is not None:
                    st.session_state.data_manager.set_data(df)
                    st.sidebar.success(f"✓ Data loaded: {df.shape}")
            except Exception as e:
                st.sidebar.warning(f"Loaded to workspace (agent can parse). Preview failed: {e}")
        else:
            st.sidebar.info("File already uploaded (skipped re-save).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 **Workspace Files**")
    files = st.session_state.data_manager.list_workspace_files()

    def _render_file_group(label, group_key):
        items = files.get(group_key, [])
        if not items:
            return
        with st.sidebar.expander(f"{label} ({len(items)})", expanded=(group_key == "data")):
            for i, meta in enumerate(items):
                name = meta["name"]
                path = Path(meta["path"])
                cols = st.columns([3, 1])
                with cols[0]:
                    st.text(name if len(name) <= 30 else name[:30] + "…")
                with cols[1]:
                    if path.exists():
                        with open(path, "rb") as fh:
                            st.download_button("📥", data=fh.read(), file_name=name, key=f"dl_{group_key}_{i}")

    _render_file_group("📊 Data Files", "data")
    _render_file_group("📈 Plots", "plots")
    _render_file_group("📦 Exports", "exports")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 **Session**")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("💾 Save"):
            saved = st.session_state.data_manager.auto_save_state()
            st.sidebar.success("Saved!" if saved else "Nothing to save")
    with c2:
        if st.button("📦 Export"):
            if st.session_state.client:
                try:
                    st.session_state.client.export_session()
                    st.sidebar.success("Exported!")
                except Exception as e:
                    st.sidebar.error(f"Export failed: {e}")
    with c3:
        if st.button("🧹 Reset Chat"):
            st.session_state.messages = []

    with st.sidebar.expander("🔍 Quick Info", expanded=False):
        if st.button("📋 Show Status"):
            if st.session_state.client:
                st.json(st.session_state.client.get_status())
        if st.button("📊 Data Summary"):
            if st.session_state.data_manager.has_data():
                st.json(st.session_state.data_manager.get_data_summary())
            else:
                st.info("No data loaded")
        if st.button("📈 Plot History"):
            hist = st.session_state.data_manager.get_plot_history()
            if hist:
                for p in hist[-10:]:
                    st.text(f"• {p.get('title','plot')[:40]}…")
            else:
                st.info("No plots yet")


# -----------------------
# Terminal output
# -----------------------
def display_terminal_output():
    if not st.session_state.show_terminal:
        return
    logs = st.session_state.log_collector.get_formatted_logs()
    if logs and logs != "No logs available":
        with st.expander("🖥️ **Agent Execution Log**", expanded=True):
            st.code(logs, language="text")


# -----------------------
# Chat area
# -----------------------
def display_chat_interface():
    st.markdown("## 🦞 **Lobster AI** – Multi-Agent Bioinformatics System")
    st.caption("by homara AI")
    st.markdown("---")

    client = init_client()

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for plot_entry in msg.get("plots", []) or []:
                if isinstance(plot_entry, dict) and "figure" in plot_entry:
                    st.plotly_chart(plot_entry["figure"], use_container_width=True)

    # Input (disabled while processing to avoid overlaps)
    new_prompt = st.chat_input("Ask about your bioinformatics data…", disabled=st.session_state.processing)

    # If user submits while processing, we simply don't accept (UI disables input).
    if new_prompt:
        # Immediately display the user message in THIS run
        st.session_state.messages.append({"role": "user", "content": new_prompt})
        with st.chat_message("user"):
            st.markdown(new_prompt)

        # Guard against double execution
        if st.session_state.processing:
            st.warning("Already processing a request, please wait…")
            return

        st.session_state.processing = True
        st.session_state.log_collector.clear()

        # Run the agent ONCE in this same run
        with st.chat_message("assistant"):
            with st.spinner("🦞 Analyzing…"):
                try:
                    result = client.query(new_prompt, stream=False)

                    if result.get("success"):
                        response = result.get("response", "")
                        st.markdown(response)

                        # Show logs after completion
                        display_terminal_output()

                        # Show plots (if any)
                        plots = st.session_state.data_manager.latest_plots
                        if plots:
                            st.markdown("---")
                            st.markdown("### 📊 Generated Visualizations")
                            for plot_entry in plots:
                                if isinstance(plot_entry, dict) and "figure" in plot_entry:
                                    if "title" in plot_entry:
                                        st.markdown(f"**{plot_entry['title']}**")
                                    st.plotly_chart(plot_entry["figure"], use_container_width=True)
                                    # Optional: download as HTML
                                    html_str = plot_entry["figure"].to_html()
                                    st.download_button(
                                        "📥 Download HTML",
                                        data=html_str,
                                        file_name=f"{plot_entry.get('id','plot')}.html",
                                        mime="text/html",
                                        key=f"dl_plot_{id(plot_entry)}",
                                    )

                        # Persist assistant message (+ plots) to history
                        save_msg = {"role": "assistant", "content": response}
                        if st.session_state.data_manager.latest_plots:
                            save_msg["plots"] = st.session_state.data_manager.latest_plots.copy()
                        st.session_state.messages.append(save_msg)

                    else:
                        err = f"❌ Error: {result.get('error','Unknown error')}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})

                except Exception as e:
                    err = f"❌ System error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    logger.exception("Agent error")

                finally:
                    st.session_state.processing = False


# -----------------------
# Main
# -----------------------
def main():
    init_session_state()
    display_sidebar()
    display_chat_interface()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;color:#666'>
        🦞 <b>Lobster AI</b> v2.0 | Multi-Agent Bioinformatics System | © 2025 homara AI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()