"""Main analysis screen with cockpit-style layout."""

from typing import Optional
from functools import partial

from textual.screen import Screen
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer
from textual.worker import Worker
from textual.binding import Binding
from textual import on

from lobster.core.client import AgentClient
from lobster.core.license_manager import get_current_tier
from lobster.config.llm_factory import LLMFactory
from lobster.config.settings import get_settings
from lobster.ui.widgets import (
    QueryPrompt,
    ModalityList,
    DataHub,
    FileLoadRequested,
    DataHubModalitySelected,
    ResultsDisplay,
    PlotPreview,
    StatusBar,
    SystemInfoPanel,
    QueuePanel,
    QueueStatusBar,
    ConnectionsPanel,
    AgentsPanel,
    AdaptersPanel,
    TokenUsagePanel,
    ActivityLogPanel,
)
from lobster.ui.widgets.status_bar import get_friendly_model_name
from lobster.ui.widgets.modality_list import ModalitySelected
from lobster.ui.callbacks import TextualCallbackHandler
from lobster.ui.services import ErrorService, ErrorCategory
from lobster.services.data_management.modality_management_service import ModalityManagementService
from lobster.cli_internal.commands import (
    DashboardOutputAdapter,
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    QueueFileTypeNotSupported,
)


class AnalysisScreen(Screen):
    """
    Cockpit-style analysis screen - information-dense dashboard.

    Layout (NASA mission control inspired):
    ┌─────────────────────────────────────────────────────────────┐
    │ Header: Lobster OS                                          │
    ├─────────────────────────────────────────────────────────────┤
    │ [tier] │ [provider/model] │ [agent status]      [status bar]│
    ├────────────────┬─────────────────────────┬──────────────────┤
    │ SYSTEM         │                         │ AGENTS           │
    │ ● CPU 45%      │     Results             │ ● supervisor     │
    │ ● MEM 8.2/16GB │     (conversation)      │ ▶ research (act) │
    │ ● GPU CUDA     │                         │ ● data_expert    │
    ├────────────────┤                         ├──────────────────┤
    │ CONNECTIONS    │                         │ MODALITIES       │
    │ ● GEO  ● SRA   │                         │ ● geo_gse12345   │
    │ ● PubMed ○PRIDE│                         │ ● rna_filtered   │
    ├────────────────┤                         ├──────────────────┤
    │ QUEUES         │                         │ ADAPTERS         │
    │ Downloads: 2   │                         │ Trans: ●●●●      │
    │ Papers: 15     ├─────────────────────────┤ Prot:  ●●●       │
    ├────────────────┤    [Query Prompt]       ├──────────────────┤
    │                │                         │ PLOTS            │
    │                │                         │ ○ No plots yet   │
    ├────────────────┴─────────────────────────┴──────────────────┤
    │ Footer: ESC Quit │ ^P Commands │ ^L Clear │ F5 Refresh      │
    └─────────────────────────────────────────────────────────────┘
    """

    CSS = """
    AnalysisScreen {
        background: transparent;
    }

    /* Main 3-column layout */
    #main-panels {
        height: 1fr;
    }

    /* Left panel - system telemetry */
    #left-panel {
        width: 35;
        border-right: solid #CC2C18 30%;
        padding: 0 1;
    }

    #left-panel > * {
        margin-bottom: 1;
    }

    /* Center panel - conversation */
    #center-panel {
        width: 1fr;
        padding: 0 1;
    }

    #results-display {
        height: 1fr;
        border: round #CC2C18 20%;
        overflow-y: auto;
        padding: 0 1;
    }

    #query-prompt {
        height: 6;
        border: round #CC2C18 40%;
        margin-top: 1;
    }

    #query-prompt:focus {
        border: round #CC2C18 80%;
    }

    /* Right panel - data & agents */
    #right-panel {
        width: 35;
        border-left: solid #CC2C18 30%;
        padding: 0 1;
    }

    #right-panel > * {
        margin-bottom: 1;
    }

    /* Cockpit panel styling - compact */
    SystemInfoPanel, ConnectionsPanel, QueuePanel, QueueStatusBar,
    AgentsPanel, AdaptersPanel, TokenUsagePanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    /* Activity log - scrollable */
    ActivityLogPanel {
        height: 8;
        max-height: 10;
        border: round #CC2C18 30%;
        padding: 0 1;
    }

    /* Data panels */
    ModalityList {
        height: 1fr;
        min-height: 6;
        border: round #CC2C18 30%;
    }

    DataHub {
        height: auto;
        min-height: 12;
        max-height: 25;
        border: round #CC2C18 30%;
    }

    PlotPreview {
        height: auto;
        max-height: 8;
        border: round #CC2C18 30%;
    }

    /* Chat messages - minimal styling */
    ChatMessage {
        height: auto;
        width: 1fr;
        margin: 0 0 1 0;
        padding: 1;
    }

    .user-message {
        border: round #4a9eff 60%;
        background: #4a9eff 10%;
    }

    .user-message Markdown {
        color: #4a9eff;
    }

    .agent-message {
        border: round #CC2C18 40%;
        background: #CC2C18 5%;
    }

    .agent-message.streaming {
        border: round #E84D3A 70%;
        background: #E84D3A 8%;
    }

    .system-message {
        border: round #888888 40%;
        background: #888888 5%;
    }

    .error-message {
        border: round #ff4444 60%;
        background: #ff4444 10%;
    }

    /* Status bar */
    StatusBar {
        height: 1;
        background: transparent;
        border-bottom: solid #CC2C18 20%;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", key_display="ESC", priority=True),
        Binding("ctrl+q", "quit", "Quit", key_display="^Q"),
        Binding("ctrl+p", "command_palette", "Commands", key_display="^P"),
        Binding("ctrl+c", "cancel_query", "Cancel", key_display="^C"),
        Binding("ctrl+l", "clear_results", "Clear", key_display="^L"),
        Binding("f5", "refresh_data", "Refresh", key_display="F5"),
    ]

    def __init__(self, client: AgentClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.current_worker: Optional[Worker] = None
        self.error_service: Optional[ErrorService] = None
        self._last_query: str = ""  # For retry recovery

    def compose(self):
        """Create cockpit-style 3-column layout."""
        yield Header()

        # Status bar - top telemetry strip
        yield StatusBar(id="status-bar")

        # Main content area - 3 columns
        with Horizontal(id="main-panels"):
            # Left panel: System telemetry
            with Vertical(id="left-panel"):
                yield SystemInfoPanel()
                yield ConnectionsPanel()
                yield TokenUsagePanel()
                yield QueuePanel(self.client)
                yield QueueStatusBar(self.client)

            # Center panel: Conversation + Activity
            with Vertical(id="center-panel"):
                yield ResultsDisplay(id="results-display")
                yield ActivityLogPanel(id="activity-log")
                yield QueryPrompt(id="query-prompt")

            # Right panel: Agents & Data
            with Vertical(id="right-panel"):
                yield AgentsPanel(client=self.client)
                yield DataHub(
                    client=self.client,
                    workspace_path=self.client.data_manager.workspace_path,
                    on_load_file=self._load_file_from_workspace,
                    id="data-hub"
                )
                yield AdaptersPanel()
                yield PlotPreview(self.client)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen."""
        self.sub_title = f"Session: {self.client.session_id}"
        self.error_service = ErrorService(self.app)
        self._init_status_bar()
        self._inject_ui_callback()

    def _init_status_bar(self) -> None:
        """Initialize status bar with client data."""
        status_bar = self.query_one(StatusBar)

        # Subscription tier
        status_bar.subscription_tier = get_current_tier()

        # Provider + Model
        try:
            provider = LLMFactory.get_current_provider()
            if provider:
                status_bar.provider_name = provider
                model_params = get_settings().get_agent_llm_params("supervisor")
                model_id = model_params.get("model_id", "unknown")
                status_bar.model_name = get_friendly_model_name(model_id, provider)
        except Exception:
            status_bar.provider_name = "unknown"
            status_bar.model_name = "unknown"

    def _inject_ui_callback(self) -> None:
        """
        Inject TextualCallbackHandler for live UI updates.

        IMPORTANT: Pass direct widget references to avoid race conditions.
        Widgets must be mounted before callbacks can update them.
        """
        try:
            # Get direct references to widgets (they're mounted by now)
            activity_log = self.query_one(ActivityLogPanel)
            agents_panel = self.query_one(AgentsPanel)
            token_panel = self.query_one(TokenUsagePanel)

            ui_callback = TextualCallbackHandler(
                app=self.app,
                activity_log=activity_log,
                agents_panel=agents_panel,
                token_panel=token_panel,
                show_reasoning=False,
                show_tools=True,
                debug=False,  # Set True to see callback events in notifications
            )
            self.client.callbacks.append(ui_callback)
        except Exception as e:
            # Log error if widgets not found
            self.notify(f"Callback setup error: {str(e)[:50]}", severity="warning")

    def _load_file_from_workspace(self, filepath: str) -> None:
        """
        Load a file from workspace into memory as a modality.

        This is the callback for DataHub's file loading.

        Args:
            filepath: Absolute path to file in workspace
        """
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)

        try:
            file_path = Path(filepath)

            # Determine adapter based on file extension
            ext = file_path.suffix.lower()

            # Map extensions to adapters
            adapter_map = {
                ".h5ad": "transcriptomics_single_cell",
                ".csv": "transcriptomics_bulk",
                ".tsv": "transcriptomics_bulk",
                ".xlsx": "transcriptomics_bulk",
                ".xls": "transcriptomics_bulk",
                ".txt": "transcriptomics_bulk",
                ".mtx": "transcriptomics_single_cell",
                ".gz": "transcriptomics_single_cell",  # Assume 10x format
            }

            adapter = adapter_map.get(ext)
            if not adapter:
                self.notify(f"Unsupported format: {ext}", severity="error")
                return

            # Generate modality name from filename (remove extension, make safe)
            modality_name = file_path.stem.replace(".", "_").replace(" ", "_").lower()

            # Ensure unique name
            existing = self.client.data_manager.list_modalities()
            if modality_name in existing:
                counter = 1
                base_name = modality_name
                while modality_name in existing:
                    modality_name = f"{base_name}_{counter}"
                    counter += 1

            # Create modality management service
            service = ModalityManagementService(self.client.data_manager)

            # Load the file
            logger.info(f"Loading {file_path.name} as '{modality_name}' using adapter '{adapter}'")

            adata, stats, ir = service.load_modality(
                modality_name=modality_name,
                file_path=str(file_path),
                adapter=adapter,
                dataset_type="workspace",
                validate=True,
            )

            # Success - update conversation display
            results = self.query_one(ResultsDisplay)
            results.append_assistant_message(
                f"Loaded {file_path.name} as '{modality_name}'\n"
                f"Shape: {stats['shape']['n_obs']:,} observations × {stats['shape']['n_vars']:,} variables"
            )

            logger.info(f"Successfully loaded {modality_name}: {stats}")

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}", exc_info=True)
            self.notify(f"Load failed: {str(e)[:60]}", severity="error")

            # Also show in conversation
            results = self.query_one(ResultsDisplay)
            results.append_system_message(f"❌ Failed to load {Path(filepath).name}: {str(e)[:100]}")

    @on(QueryPrompt.QuerySubmitted)
    def handle_query_submission(self, event: QueryPrompt.QuerySubmitted) -> None:
        """Handle query submission."""
        query_text = event.text

        # Validate input
        if not query_text.strip():
            if self.error_service:
                self.error_service.show_validation_error("Cannot send empty message!")
            return

        # Handle slash commands
        if query_text.strip().startswith("/"):
            self._handle_slash_command(query_text.strip())
            return

        # Store for retry recovery
        self._last_query = query_text

        # Show user message
        results = self.query_one(ResultsDisplay)
        results.append_user_message(query_text)

        # Update status bar
        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "processing"

        # Lock input
        event.prompt_input.submit_ready = False

        # Run query in background
        self.run_worker(
            partial(self.execute_streaming_query, query_text),
            name=f"query_{query_text[:20]}",
            group="agent_query",
            exclusive=True,
            thread=True,
        )

    def _handle_slash_command(self, cmd: str) -> None:
        """Handle slash commands in the dashboard."""
        results = self.query_one(ResultsDisplay)

        if cmd in ("/help", "/h", "/?"):
            help_text = """**Dashboard Commands:**

**General:**
- `/help` - Show this help
- `/session` - Show session status
- `/status` - Show tier, packages, and agents
- `/clear` - Clear conversation
- `/exit` - Exit dashboard

**Data:**
- `/data` - Show loaded modalities
- `/plots` - Refresh plots panel
- `/save` - Save plots to workspace

**Queue:**
- `/queue` - Show queue status
- `/queue list` - List queue entries
- `/queue load <file>` - Load .ris file into queue
- `/queue clear [download|all]` - Clear queue(s)
- `/queue export [name]` - Export queue to workspace

**Navigation:**
- Use **arrow keys** in panels to navigate
- Press **Enter** to select/open items
- Press **F5** to refresh all panels
- Press **Ctrl+L** to clear results
- Press **ESC** to quit"""
            results.append_system_message(help_text)

        elif cmd == "/session":
            status = self.client.get_status()
            from lobster.config.agent_config import get_agent_configurator
            from lobster.config.llm_factory import LLMFactory

            configurator = get_agent_configurator()
            current_mode = configurator.get_current_profile()
            provider = LLMFactory.get_current_provider() or "unknown"

            status_text = f"""**Session Status:**
- Session ID: `{status.get('session_id', 'N/A')}`
- Mode: {current_mode}
- Messages: {status.get('message_count', 0)}
- Provider: {provider}
- Modalities: {len(status.get('modalities', []))}
- Plots: {len(self.client.data_manager.latest_plots)}
- Workspace: `{self.client.data_manager.workspace_path}`"""

            # Add data summary if available
            if status.get('has_data') and status.get('data_summary'):
                summary = status['data_summary']
                status_text += f"\n- Data shape: {summary.get('shape', 'N/A')}"
                status_text += f"\n- Memory usage: {summary.get('memory_usage', 'N/A')}"

            results.append_system_message(status_text)

        elif cmd == "/status":
            from lobster.core.license_manager import get_entitlement_status
            from lobster.core.plugin_loader import get_installed_packages
            from lobster.config.agent_registry import get_worker_agents
            from lobster.config.subscription_tiers import is_agent_available

            # Get entitlement status
            try:
                entitlement = get_entitlement_status()
            except ImportError:
                entitlement = {"tier": "free", "tier_display": "Free", "source": "default"}

            # Get installed packages
            try:
                packages = get_installed_packages()
            except ImportError:
                packages = {"lobster-ai": "unknown"}

            # Get available agents
            try:
                worker_agents = get_worker_agents()
                tier = entitlement.get("tier", "free")
                available = [name for name in worker_agents if is_agent_available(name, tier)]
                restricted = [name for name in worker_agents if not is_agent_available(name, tier)]
            except ImportError:
                available = []
                restricted = []

            tier_display = entitlement.get("tier_display", "Free")
            tier_emoji = {"free": "🆓", "premium": "⭐", "enterprise": "🏢"}.get(
                entitlement.get("tier", "free"), "🆓"
            )

            status_text = f"""**Installation Status:**

**Subscription:**
- Tier: {tier_emoji} {tier_display}
- Source: {entitlement.get('source', 'default')}"""

            if entitlement.get("expires_at"):
                days = entitlement.get("days_until_expiry")
                if days is not None and days < 30:
                    status_text += f"\n- ⚠️ Expires in {days} days"

            status_text += f"\n\n**Packages ({len(packages)}):**"
            for pkg_name, version in list(packages.items())[:5]:  # Show first 5
                icon = "✓" if version not in ["missing", "dev"] else ("✗" if version == "missing" else "⚡")
                status_text += f"\n- {icon} {pkg_name}: {version}"
            if len(packages) > 5:
                status_text += f"\n- ... and {len(packages) - 5} more"

            if available:
                status_text += f"\n\n**Available Agents ({len(available)}):**"
                for agent in sorted(available)[:5]:
                    status_text += f"\n- {agent}"
                if len(available) > 5:
                    status_text += f"\n- ... and {len(available) - 5} more"

            if restricted:
                status_text += f"\n\n**Premium Agents ({len(restricted)}):** (upgrade required)"
                for agent in sorted(restricted)[:3]:
                    status_text += f"\n- {agent}"
                if len(restricted) > 3:
                    status_text += f"\n- ... and {len(restricted) - 3} more"

            results.append_system_message(status_text)

        elif cmd == "/data":
            modalities = list(self.client.data_manager.modalities.keys())
            if modalities:
                mod_list = "\n".join(f"- `{m}`" for m in modalities)
                results.append_system_message(f"**Loaded Modalities:**\n{mod_list}")
            else:
                results.append_system_message("*No modalities loaded*")
            # Also refresh the data hub panel
            self.query_one(DataHub).refresh_loaded()

        elif cmd == "/plots":
            self.query_one(PlotPreview).refresh_plots()
            plot_count = len(self.client.data_manager.latest_plots)
            results.append_system_message(f"Plots refreshed. {plot_count} plots available.")

        elif cmd == "/clear":
            results.clear_display()
            self.notify("Cleared", timeout=1)

        elif cmd == "/save":
            saved = self.client.data_manager.save_plots_to_workspace()
            if saved:
                results.append_system_message(f"Saved {len(saved)} plots to workspace.")
                self.query_one(PlotPreview).refresh_plots()
            else:
                results.append_system_message("No plots to save.")

        elif cmd in ("/exit", "/quit", "/q"):
            self.app.exit()

        elif cmd.startswith("/plot "):
            # Open specific plot
            plot_id = cmd.replace("/plot ", "").strip()
            plots = self.client.data_manager.latest_plots
            for i, p in enumerate(plots):
                if p.get("id") == plot_id or plot_id in p.get("original_title", ""):
                    self.query_one(PlotPreview)._open_plot(i)
                    return
            self.notify(f"Plot not found: {plot_id}", severity="warning")

        # Queue commands (shared implementation with CLI)
        elif cmd.startswith("/queue"):
            output = DashboardOutputAdapter(results)
            parts = cmd.split(maxsplit=2)
            subcommand = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None

            try:
                if not subcommand or subcommand == "status":
                    # /queue or /queue status - show queue status
                    show_queue_status(self.client, output)

                elif subcommand == "list":
                    # /queue list - list all queue entries
                    queue_list(self.client, output)

                elif subcommand == "load":
                    # /queue load <file> - load .ris file into queue
                    if not arg:
                        results.append_system_message("Usage: /queue load <file>")
                    else:
                        queue_load_file(self.client, arg, output, current_directory=None)

                elif subcommand == "clear":
                    # /queue clear [download|all] - clear queue(s)
                    queue_type = arg if arg in ["download", "all"] else "publication"
                    queue_clear(self.client, output, queue_type)

                elif subcommand == "export":
                    # /queue export [name] - export queue to workspace
                    queue_export(self.client, arg, output)

                else:
                    results.append_system_message(
                        f"Unknown queue subcommand: {subcommand}\n"
                        "Available: status, list, load, clear, export"
                    )

            except QueueFileTypeNotSupported as e:
                results.append_system_message(f"❌ {str(e)}")
            except Exception as e:
                results.append_system_message(f"❌ Queue command failed: {str(e)}")

        else:
            results.append_system_message(
                f"*Command `{cmd}` not available in dashboard. Use CLI mode for full commands.*\n\n"
                "Type `/help` for available dashboard commands."
            )

    def execute_streaming_query(self, query: str) -> None:
        """Execute query with streaming and professional error handling."""
        results = self.query_one(ResultsDisplay)

        try:
            self.app.call_from_thread(results.start_agent_message)

            for event in self.client.query(query, stream=True):
                event_type = event.get("type")

                if event_type == "stream":
                    content = event.get("content", "")
                    if content:
                        # Filter out handoff/transfer messages from display
                        if self._is_handoff_message(content):
                            continue
                        self.app.call_from_thread(
                            results.append_to_agent_message, content
                        )
                elif event_type == "complete":
                    self.app.call_from_thread(results.complete_agent_message)
                elif event_type == "error":
                    error = event.get("error", "Unknown error")
                    self.app.call_from_thread(
                        self._handle_query_error, error, query
                    )

        except ConnectionError as e:
            self.app.call_from_thread(
                self._handle_connection_error, str(e), query
            )
        except TimeoutError as e:
            self.app.call_from_thread(
                self._handle_connection_error, f"Request timed out: {e}", query
            )
        except Exception as e:
            self.app.call_from_thread(
                self._handle_query_error, str(e), query
            )
        finally:
            self.app.call_from_thread(self._unlock_input)

    def _handle_query_error(self, error_msg: str, original_query: str) -> None:
        """Handle query execution error with recovery option."""
        results = self.query_one(ResultsDisplay)
        results.show_error(error_msg)

        if self.error_service:
            self.error_service.handle_error(
                Exception(error_msg),
                category=ErrorCategory.AGENT,
                context="Query execution",
                show_modal=False,
            )

        # Restore query to prompt for easy retry
        try:
            prompt = self.query_one(QueryPrompt)
            if hasattr(prompt, 'input_widget'):
                prompt.input_widget.value = original_query
        except Exception:
            pass

    def _handle_connection_error(self, error_msg: str, original_query: str) -> None:
        """Handle connection error with modal and retry option."""
        results = self.query_one(ResultsDisplay)
        results.show_error(f"Connection error: {error_msg}")

        if self.error_service:
            def retry_query():
                # Re-submit the query
                self.app.call_from_thread(
                    self._retry_last_query
                )

            self.error_service.show_connection_error(
                Exception(error_msg),
                provider="LLM Provider",
                can_retry=True,
                on_retry=retry_query,
            )

    def _retry_last_query(self) -> None:
        """Retry the last failed query."""
        if self._last_query:
            # Re-run the query
            self.run_worker(
                partial(self.execute_streaming_query, self._last_query),
                name=f"retry_{self._last_query[:20]}",
                group="agent_query",
                exclusive=True,
                thread=True,
            )

    def _unlock_input(self) -> None:
        """Unlock input and refresh views."""
        prompt = self.query_one(QueryPrompt)
        prompt.submit_ready = True

        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "idle"

        # Mark all agents as idle - direct widget update since we're in main thread
        try:
            agents_panel = self.query_one(AgentsPanel)
            agents_panel.set_all_idle()
        except Exception:
            pass

        # Reset callback tracking state
        for callback in self.client.callbacks:
            if isinstance(callback, TextualCallbackHandler):
                callback.current_agent = None
                callback.agent_stack = []
                break

        # Log completion in activity log
        try:
            activity_log = self.query_one(ActivityLogPanel)
            activity_log.log_complete("Query complete")
        except Exception:
            pass

        # Refresh data panels
        self.query_one(DataHub).refresh_all()
        self.query_one(PlotPreview).refresh_plots()

    def _is_handoff_message(self, content: str) -> bool:
        """Check if content is a handoff/transfer message that should be filtered."""
        if not content:
            return False
        content_lower = content.lower().strip()
        # Filter patterns for agent handoff messages
        handoff_patterns = [
            "transfer back to supervisor",
            "transferring back to supervisor",
            "transferring to supervisor",
            "handoff to",
            "handing off to",
            "delegating to",
        ]
        return any(pattern in content_lower for pattern in handoff_patterns)

    def on_modality_list_modality_selected(self, event: ModalitySelected) -> None:
        """Handle modality selection from legacy ModalityList (backward compatibility)."""
        self.notify(f"Selected: {event.modality_name}", timeout=2)

    def on_data_hub_modality_selected(self, event: DataHubModalitySelected) -> None:
        """Handle modality selection from DataHub."""
        self.notify(f"Selected: {event.modality_name}", timeout=2)
        # TODO: Future enhancement - show modality details modal

    def refresh_all(self) -> None:
        """Refresh all panels (called by F5)."""
        self.query_one(DataHub).refresh_all()
        self.query_one(PlotPreview).refresh_plots()

    def action_cancel_query(self) -> None:
        """Cancel running query."""
        if self.current_worker and not self.current_worker.is_finished:
            self.current_worker.cancel()
            self.notify("Cancelled", severity="warning")
            self._unlock_input()

    def action_clear_results(self) -> None:
        """Clear results display."""
        self.query_one(ResultsDisplay).clear_display()

    def action_refresh_data(self) -> None:
        """Refresh all data panels."""
        self.refresh_all()
        self.notify("Refreshed", timeout=1)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
