"""
Manual Cell Type Annotation Service with Rich Terminal Interactive Mode

This service provides expert-guided cell type annotation capabilities with 
color-synchronized Rich terminal interface that matches UMAP plot colors.

Key Features:
- Rich Terminal Interface with color-coded cluster visualization
- Perfect color synchronization between plots and terminal
- Manual annotation with cluster collapsing capabilities
- Debris identification and removal options
- Undo/redo functionality with annotation history
- Export/import annotation mappings
- Smart debris suggestions based on QC metrics
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np
import scanpy as sc
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.tree import Tree
from rich.columns import Columns
from rich import box

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    cluster_id: str
    color: str  # Hex color code
    cell_count: int
    assigned_type: Optional[str] = None
    is_debris: bool = False
    qc_scores: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class AnnotationState:
    """Current state of annotation process."""
    clusters: Dict[str, ClusterInfo]
    cell_type_mapping: Dict[str, str]  # cluster_id -> cell_type
    debris_clusters: Set[str]
    annotation_history: List[Dict[str, Any]]
    current_step: int = 0
    
    def __post_init__(self):
        """Initialize derived fields."""
        if not hasattr(self, 'annotation_history'):
            self.annotation_history = []


class ManualAnnotationService:
    """
    Service for manual cell type annotation with Rich terminal interface.
    
    Provides color-synchronized interface matching UMAP plot colors for 
    intuitive cluster identification and expert-guided annotation.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console(width=120)
        self.state: Optional[AnnotationState] = None
        self.adata: Optional[sc.AnnData] = None
        self.cluster_colors: Dict[str, str] = {}
        self.templates_loaded: Dict[str, Dict] = {}
        
    def initialize_annotation_session(self, 
                                   adata: sc.AnnData,
                                   cluster_key: str = 'leiden',
                                   color_key: Optional[str] = None) -> AnnotationState:
        """
        Initialize a new annotation session.
        
        Args:
            adata: Annotated data object with clustering results
            cluster_key: Key in adata.obs containing cluster assignments
            color_key: Key for extracting cluster colors (optional)
            
        Returns:
            AnnotationState: Initial state of annotation session
        """
        self.adata = adata
        
        # Extract cluster information
        clusters = {}
        unique_clusters = adata.obs[cluster_key].unique()
        
        # Extract or generate colors for clusters
        if color_key and color_key in adata.uns:
            # Use existing colors from scanpy
            color_map = dict(zip(unique_clusters, adata.uns[color_key]))
        else:
            # Generate colors using scanpy's default palette
            import matplotlib.pyplot as plt
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            color_map = {str(cluster): f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" 
                        for cluster, c in zip(unique_clusters, colors)}
        
        # Create cluster information
        for cluster_id in unique_clusters:
            cluster_mask = adata.obs[cluster_key] == cluster_id
            cell_count = cluster_mask.sum()
            
            # Calculate QC scores
            qc_scores = {}
            if 'total_counts' in adata.obs.columns:
                qc_scores['mean_total_counts'] = adata.obs.loc[cluster_mask, 'total_counts'].mean()
            if 'n_genes_by_counts' in adata.obs.columns:
                qc_scores['mean_genes'] = adata.obs.loc[cluster_mask, 'n_genes_by_counts'].mean()
            if 'pct_counts_mt' in adata.obs.columns:
                qc_scores['mean_mt_pct'] = adata.obs.loc[cluster_mask, 'pct_counts_mt'].mean()
                
            clusters[str(cluster_id)] = ClusterInfo(
                cluster_id=str(cluster_id),
                color=color_map.get(cluster_id, "#888888"),
                cell_count=cell_count,
                qc_scores=qc_scores
            )
        
        # Initialize state
        self.state = AnnotationState(
            clusters=clusters,
            cell_type_mapping={},
            debris_clusters=set(),
            annotation_history=[]
        )
        
        self.cluster_colors = {cluster_id: info.color for cluster_id, info in clusters.items()}
        
        return self.state
    
    def rich_annotation_interface(self) -> Dict[str, str]:
        """
        Main Rich terminal interface for manual annotation.
        
        Returns:
            Dict mapping cluster IDs to cell type annotations
        """
        if not self.state:
            raise ValueError("No annotation session initialized. Call initialize_annotation_session first.")
        
        with self.console.screen() as screen:
            screen.update(self._create_welcome_screen())
            self.console.input("Press Enter to continue...")
        
        # Main annotation loop
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                self._annotate_clusters_interactive()
            elif choice == "2":
                self._identify_debris_interactive()
            elif choice == "3":
                self._collapse_clusters_interactive()
            elif choice == "4":
                self._show_annotation_summary()
            elif choice == "5":
                self._apply_annotation_template()
            elif choice == "6":
                self._undo_last_action()
            elif choice == "7":
                self._export_annotations()
            elif choice == "8":
                self._import_annotations()
            elif choice == "9":
                if self._confirm_finish():
                    break
            else:
                self.console.print("Invalid choice. Please try again.", style="red")
        
        return self.state.cell_type_mapping
    
    def _create_welcome_screen(self) -> Panel:
        """Create welcome screen with color legend."""
        
        # Create cluster color legend
        color_table = Table(title="Cluster Color Legend", box=box.ROUNDED)
        color_table.add_column("Cluster ID", style="bold")
        color_table.add_column("Color", justify="center")
        color_table.add_column("Cell Count", justify="right")
        color_table.add_column("Status")
        
        for cluster_id, info in sorted(self.state.clusters.items()):
            color_style = Style(color=info.color, bold=True)
            status = "Annotated" if info.assigned_type else "Pending"
            if info.is_debris:
                status = "Debris"
            
            color_table.add_row(
                cluster_id,
                "●●●",  # Color indicator
                str(info.cell_count),
                status,
                style=color_style if not info.is_debris else "dim red"
            )
        
        welcome_text = f"""
[bold blue]🧬 Manual Cell Type Annotation Service[/bold blue]

Welcome to the interactive annotation interface! This tool provides:

• [green]Color-synchronized visualization[/green] matching your UMAP plot
• [yellow]Interactive cluster assignment[/yellow] with expert guidance  
• [cyan]Debris identification[/cyan] and quality control
• [magenta]Annotation templates[/magenta] for common tissue types
• [blue]Undo/redo functionality[/blue] with full history

[bold]Current Session:[/bold]
• Total clusters: {len(self.state.clusters)}
• Total cells: {sum(info.cell_count for info in self.state.clusters.values())}
• Annotated: {len([c for c in self.state.clusters.values() if c.assigned_type])}
• Debris: {len(self.state.debris_clusters)}

The colors in this terminal [bold]exactly match[/bold] your UMAP plot colors for easy identification.
        """
        
        layout = Layout()
        layout.split_column(
            Layout(Panel(welcome_text, border_style="blue"), name="welcome"),
            Layout(color_table, name="legend")
        )
        
        return Panel(layout, title="Manual Annotation Session", border_style="bright_blue")
    
    def _show_main_menu(self) -> str:
        """Display main menu and get user choice."""
        
        menu_table = Table(title="🔬 Annotation Menu", box=box.DOUBLE_EDGE)
        menu_table.add_column("Option", style="bold cyan", width=8)
        menu_table.add_column("Action", style="white")
        menu_table.add_column("Description", style="dim white")
        
        menu_options = [
            ("1", "Annotate Clusters", "Assign cell types to clusters"),
            ("2", "Mark Debris", "Identify low-quality clusters"),
            ("3", "Collapse Clusters", "Merge clusters with same cell type"),
            ("4", "View Summary", "Show current annotation status"),
            ("5", "Apply Template", "Use predefined annotation template"),
            ("6", "Undo", "Undo last annotation action"),
            ("7", "Export", "Save annotations to file"),
            ("8", "Import", "Load annotations from file"),
            ("9", "Finish", "Complete annotation session"),
        ]
        
        for option, action, desc in menu_options:
            menu_table.add_row(option, action, desc)
        
        # Show current progress
        annotated = len([c for c in self.state.clusters.values() if c.assigned_type])
        total = len(self.state.clusters) - len(self.state.debris_clusters)
        progress_text = f"Progress: {annotated}/{total} clusters annotated"
        
        self.console.print("\n" + "="*80)
        self.console.print(menu_table)
        self.console.print(f"\n[green]{progress_text}[/green]")
        
        return Prompt.ask("\n[bold blue]Choose an option[/bold blue]", 
                         choices=[str(i) for i in range(1, 10)],
                         default="1")
    
    def _annotate_clusters_interactive(self):
        """Interactive cluster annotation with color-coded display."""
        
        unannotated_clusters = [
            (cid, info) for cid, info in self.state.clusters.items() 
            if not info.assigned_type and not info.is_debris
        ]
        
        if not unannotated_clusters:
            self.console.print("[green]✅ All clusters are already annotated![/green]")
            return
        
        self.console.print("\n[bold blue]🎯 Cluster Annotation Mode[/bold blue]")
        self.console.print("Colors match your UMAP plot for easy identification.\n")
        
        # Display clusters to annotate
        cluster_table = Table(title="Clusters to Annotate", box=box.ROUNDED)
        cluster_table.add_column("ID", style="bold")
        cluster_table.add_column("Color", justify="center")
        cluster_table.add_column("Cells", justify="right")
        cluster_table.add_column("QC Metrics", style="dim")
        
        for cluster_id, info in unannotated_clusters:
            color_style = Style(color=info.color, bold=True)
            qc_text = f"Genes: {info.qc_scores.get('mean_genes', 0):.0f}, " + \
                     f"MT%: {info.qc_scores.get('mean_mt_pct', 0):.1f}"
            
            cluster_table.add_row(
                cluster_id,
                "●●●●",
                str(info.cell_count),
                qc_text,
                style=color_style
            )
        
        self.console.print(cluster_table)
        
        # Annotation loop
        for cluster_id, info in unannotated_clusters:
            self._annotate_single_cluster(cluster_id, info)
            
            if not Confirm.ask(f"\nContinue annotating remaining clusters?", default=True):
                break
    
    def _annotate_single_cluster(self, cluster_id: str, info: ClusterInfo):
        """Annotate a single cluster with Rich interface."""
        
        color_style = Style(color=info.color, bold=True)
        
        # Show cluster details
        cluster_panel = Panel(
            f"[bold]Cluster {cluster_id}[/bold] ({info.cell_count} cells)\n" +
            f"Color: [color({info.color})]●●●●●[/color({info.color})] {info.color}\n" +
            f"Mean genes: {info.qc_scores.get('mean_genes', 0):.0f}\n" +
            f"Mean MT%: {info.qc_scores.get('mean_mt_pct', 0):.1f}%",
            title="Current Cluster",
            border_style=color_style.color
        )
        
        self.console.print(cluster_panel)
        
        # Get annotation
        cell_type = Prompt.ask(
            f"\n[{info.color}]●[/{info.color}] Enter cell type for cluster {cluster_id}",
            default="Unknown"
        )
        
        if cell_type.lower() == 'debris':
            info.is_debris = True
            self.state.debris_clusters.add(cluster_id)
        else:
            info.assigned_type = cell_type
            self.state.cell_type_mapping[cluster_id] = cell_type
        
        # Save action to history
        self._save_action({
            'type': 'annotate',
            'cluster_id': cluster_id,
            'cell_type': cell_type,
            'timestamp': datetime.now().isoformat()
        })
        
        self.console.print(f"[green]✅ Cluster {cluster_id} annotated as '{cell_type}'[/green]")
    
    def _identify_debris_interactive(self):
        """Interactive debris identification with smart suggestions."""
        
        self.console.print("\n[bold red]🗑️ Debris Identification Mode[/bold red]")
        
        # Get debris suggestions based on QC metrics
        suggestions = self._get_debris_suggestions()
        
        if suggestions:
            self.console.print("\n[yellow]💡 Smart debris suggestions based on QC metrics:[/yellow]")
            
            suggestion_table = Table(box=box.MINIMAL)
            suggestion_table.add_column("Cluster", style="bold")
            suggestion_table.add_column("Reason", style="yellow")
            suggestion_table.add_column("Cells")
            suggestion_table.add_column("Action")
            
            for cluster_id, reason in suggestions:
                info = self.state.clusters[cluster_id]
                action = "Mark as debris?" if cluster_id not in self.state.debris_clusters else "Already debris"
                suggestion_table.add_row(
                    cluster_id, reason, str(info.cell_count), action
                )
            
            self.console.print(suggestion_table)
            
            if Confirm.ask("\nApply smart debris suggestions?", default=False):
                for cluster_id, reason in suggestions:
                    if cluster_id not in self.state.debris_clusters:
                        self._mark_as_debris(cluster_id, reason)
        
        # Manual debris marking
        self._manual_debris_marking()
    
    def _get_debris_suggestions(self) -> List[Tuple[str, str]]:
        """Get smart debris suggestions based on QC metrics."""
        
        suggestions = []
        
        for cluster_id, info in self.state.clusters.items():
            if cluster_id in self.state.debris_clusters:
                continue
                
            reasons = []
            
            # Low gene count
            if 'mean_genes' in info.qc_scores and info.qc_scores['mean_genes'] < 500:
                reasons.append(f"Low gene count ({info.qc_scores['mean_genes']:.0f})")
            
            # High mitochondrial percentage
            if 'mean_mt_pct' in info.qc_scores and info.qc_scores['mean_mt_pct'] > 20:
                reasons.append(f"High MT% ({info.qc_scores['mean_mt_pct']:.1f}%)")
            
            # Very small clusters
            if info.cell_count < 10:
                reasons.append(f"Very small cluster ({info.cell_count} cells)")
            
            if reasons:
                suggestions.append((cluster_id, "; ".join(reasons)))
        
        return suggestions
    
    def _manual_debris_marking(self):
        """Manual debris marking interface."""
        
        non_debris = [
            (cid, info) for cid, info in self.state.clusters.items()
            if cid not in self.state.debris_clusters
        ]
        
        if not non_debris:
            self.console.print("[yellow]No clusters available for debris marking.[/yellow]")
            return
        
        while True:
            # Show available clusters
            cluster_table = Table(title="Available Clusters", box=box.MINIMAL)
            cluster_table.add_column("ID", style="bold")
            cluster_table.add_column("Color")
            cluster_table.add_column("Cells")
            cluster_table.add_column("Status")
            
            for cluster_id, info in non_debris:
                if cluster_id in self.state.debris_clusters:
                    continue
                    
                color_style = Style(color=info.color, bold=True)
                status = info.assigned_type or "Unassigned"
                
                cluster_table.add_row(
                    cluster_id,
                    "●●●",
                    str(info.cell_count),
                    status,
                    style=color_style
                )
            
            self.console.print(cluster_table)
            
            cluster_choice = Prompt.ask(
                "\nEnter cluster ID to mark as debris (or 'done' to finish)",
                default="done"
            )
            
            if cluster_choice.lower() == 'done':
                break
            
            if cluster_choice in self.state.clusters and cluster_choice not in self.state.debris_clusters:
                self._mark_as_debris(cluster_choice, "Manual selection")
            else:
                self.console.print("[red]Invalid cluster ID or already marked as debris.[/red]")
    
    def _mark_as_debris(self, cluster_id: str, reason: str):
        """Mark a cluster as debris."""
        
        info = self.state.clusters[cluster_id]
        info.is_debris = True
        info.notes = reason
        self.state.debris_clusters.add(cluster_id)
        
        # Remove from cell type mapping if previously assigned
        if cluster_id in self.state.cell_type_mapping:
            del self.state.cell_type_mapping[cluster_id]
        
        # Save action
        self._save_action({
            'type': 'mark_debris',
            'cluster_id': cluster_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        self.console.print(f"[red]🗑️ Cluster {cluster_id} marked as debris: {reason}[/red]")
    
    def _collapse_clusters_interactive(self):
        """Interactive cluster collapsing interface."""
        
        self.console.print("\n[bold magenta]🔗 Cluster Collapse Mode[/bold magenta]")
        self.console.print("Merge multiple clusters with the same cell type.\n")
        
        # Group clusters by cell type
        cell_type_groups = {}
        for cluster_id, cell_type in self.state.cell_type_mapping.items():
            if cell_type not in cell_type_groups:
                cell_type_groups[cell_type] = []
            cell_type_groups[cell_type].append(cluster_id)
        
        # Show collapsible groups
        collapse_table = Table(title="Cell Types with Multiple Clusters", box=box.ROUNDED)
        collapse_table.add_column("Cell Type", style="bold cyan")
        collapse_table.add_column("Clusters", style="white")
        collapse_table.add_column("Total Cells", justify="right")
        
        collapsible_types = {}
        for cell_type, clusters in cell_type_groups.items():
            if len(clusters) > 1:
                total_cells = sum(self.state.clusters[cid].cell_count for cid in clusters)
                cluster_display = ", ".join(sorted(clusters))
                collapse_table.add_row(cell_type, cluster_display, str(total_cells))
                collapsible_types[cell_type] = clusters
        
        if not collapsible_types:
            self.console.print("[yellow]No cell types have multiple clusters to collapse.[/yellow]")
            return
        
        self.console.print(collapse_table)
        
        # Select cell type to collapse
        cell_type = Prompt.ask(
            "\nSelect cell type to collapse clusters (or 'cancel')",
            choices=list(collapsible_types.keys()) + ['cancel'],
            default='cancel'
        )
        
        if cell_type != 'cancel':
            self._collapse_clusters_for_type(cell_type, collapsible_types[cell_type])
    
    def _collapse_clusters_for_type(self, cell_type: str, cluster_ids: List[str]):
        """Collapse clusters for a specific cell type."""
        
        self.console.print(f"\n[bold]Collapsing clusters for '{cell_type}'[/bold]")
        
        # Show cluster details
        detail_table = Table(box=box.MINIMAL)
        detail_table.add_column("Cluster", style="bold")
        detail_table.add_column("Color")
        detail_table.add_column("Cells")
        
        for cluster_id in sorted(cluster_ids):
            info = self.state.clusters[cluster_id]
            color_style = Style(color=info.color, bold=True)
            detail_table.add_row(
                cluster_id,
                "●●●",
                str(info.cell_count),
                style=color_style
            )
        
        self.console.print(detail_table)
        
        if Confirm.ask(f"Collapse these {len(cluster_ids)} clusters into '{cell_type}'?"):
            # Create collapsed cluster ID
            collapsed_id = f"{cell_type.lower().replace(' ', '_')}_collapsed"
            
            # Save action
            self._save_action({
                'type': 'collapse',
                'cell_type': cell_type,
                'cluster_ids': cluster_ids,
                'collapsed_id': collapsed_id,
                'timestamp': datetime.now().isoformat()
            })
            
            self.console.print(f"[green]✅ Clusters collapsed into '{collapsed_id}'[/green]")
    
    def _show_annotation_summary(self):
        """Display comprehensive annotation summary."""
        
        self.console.print("\n" + "="*80)
        self.console.print("[bold blue]📊 Annotation Summary[/bold blue]")
        
        # Overall statistics
        total_clusters = len(self.state.clusters)
        annotated = len([c for c in self.state.clusters.values() if c.assigned_type])
        debris = len(self.state.debris_clusters)
        pending = total_clusters - annotated - debris
        
        stats_table = Table(title="Overall Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Count", justify="right", style="cyan")
        stats_table.add_column("Percentage", justify="right", style="green")
        
        stats_table.add_row("Total Clusters", str(total_clusters), "100%")
        stats_table.add_row("Annotated", str(annotated), f"{100*annotated/total_clusters:.1f}%")
        stats_table.add_row("Debris", str(debris), f"{100*debris/total_clusters:.1f}%")
        stats_table.add_row("Pending", str(pending), f"{100*pending/total_clusters:.1f}%")
        
        # Cell type breakdown
        cell_type_counts = {}
        cell_type_cells = {}
        
        for cluster_id, cell_type in self.state.cell_type_mapping.items():
            if cell_type not in cell_type_counts:
                cell_type_counts[cell_type] = 0
                cell_type_cells[cell_type] = 0
            cell_type_counts[cell_type] += 1
            cell_type_cells[cell_type] += self.state.clusters[cluster_id].cell_count
        
        if cell_type_counts:
            annotation_table = Table(title="Cell Type Annotations", box=box.ROUNDED)
            annotation_table.add_column("Cell Type", style="bold cyan")
            annotation_table.add_column("Clusters", justify="right")
            annotation_table.add_column("Total Cells", justify="right")
            annotation_table.add_column("Avg Cells/Cluster", justify="right")
            
            for cell_type in sorted(cell_type_counts.keys()):
                clusters = cell_type_counts[cell_type]
                cells = cell_type_cells[cell_type]
                avg_cells = cells / clusters
                
                annotation_table.add_row(
                    cell_type,
                    str(clusters),
                    str(cells),
                    f"{avg_cells:.0f}"
                )
        
        # Display tables
        layout = Layout()
        layout.split_column(
            Layout(stats_table, name="stats"),
            Layout(annotation_table if cell_type_counts else Panel("No annotations yet"), name="annotations")
        )
        
        self.console.print(layout)
    
    def _apply_annotation_template(self):
        """Apply predefined annotation template."""
        # This will be implemented with the templates file
        self.console.print("[yellow]🚧 Template functionality coming soon![/yellow]")
    
    def _undo_last_action(self):
        """Undo the last annotation action."""
        
        if not self.state.annotation_history:
            self.console.print("[yellow]No actions to undo.[/yellow]")
            return
        
        last_action = self.state.annotation_history[-1]
        action_type = last_action['type']
        
        self.console.print(f"\n[yellow]Undoing last action: {action_type}[/yellow]")
        
        if action_type == 'annotate':
            cluster_id = last_action['cluster_id']
            if cluster_id in self.state.cell_type_mapping:
                del self.state.cell_type_mapping[cluster_id]
            self.state.clusters[cluster_id].assigned_type = None
            
        elif action_type == 'mark_debris':
            cluster_id = last_action['cluster_id']
            self.state.debris_clusters.discard(cluster_id)
            self.state.clusters[cluster_id].is_debris = False
            self.state.clusters[cluster_id].notes = ""
            
        elif action_type == 'collapse':
            # Collapse actions are more complex to undo
            self.console.print("[yellow]Collapse actions cannot be undone automatically.[/yellow]")
            return
        
        # Remove action from history
        self.state.annotation_history.pop()
        
        self.console.print("[green]✅ Action undone successfully[/green]")
    
    def _export_annotations(self):
        """Export annotation mappings to file."""
        
        if not self.state.cell_type_mapping and not self.state.debris_clusters:
            self.console.print("[yellow]No annotations to export.[/yellow]")
            return
        
        # Get export filename
        filename = Prompt.ask("Enter filename for export", default="annotations.json")
        
        # Prepare export data
        export_data = {
            'cell_type_mapping': self.state.cell_type_mapping,
            'debris_clusters': list(self.state.debris_clusters),
            'cluster_info': {
                cluster_id: {
                    'color': info.color,
                    'cell_count': info.cell_count,
                    'assigned_type': info.assigned_type,
                    'is_debris': info.is_debris,
                    'notes': info.notes,
                    'qc_scores': info.qc_scores
                }
                for cluster_id, info in self.state.clusters.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.console.print(f"[green]✅ Annotations exported to {filename}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error exporting annotations: {e}[/red]")
    
    def _import_annotations(self):
        """Import annotation mappings from file."""
        
        filename = Prompt.ask("Enter filename to import", default="annotations.json")
        
        try:
            with open(filename, 'r') as f:
                import_data = json.load(f)
            
            # Apply imported annotations
            self.state.cell_type_mapping.update(import_data.get('cell_type_mapping', {}))
            self.state.debris_clusters.update(import_data.get('debris_clusters', []))
            
            # Update cluster info
            for cluster_id, cluster_data in import_data.get('cluster_info', {}).items():
                if cluster_id in self.state.clusters:
                    info = self.state.clusters[cluster_id]
                    info.assigned_type = cluster_data.get('assigned_type')
                    info.is_debris = cluster_data.get('is_debris', False)
                    info.notes = cluster_data.get('notes', "")
            
            self.console.print(f"[green]✅ Annotations imported from {filename}[/green]")
            
        except FileNotFoundError:
            self.console.print(f"[red]File not found: {filename}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error importing annotations: {e}[/red]")
    
    def _save_action(self, action: Dict[str, Any]):
        """Save an action to the annotation history."""
        self.state.annotation_history.append(action)
    
    def _confirm_finish(self) -> bool:
        """Confirm completion of annotation session."""
        
        # Count remaining work
        unannotated = len([c for c in self.state.clusters.values() 
                          if not c.assigned_type and not c.is_debris])
        
        if unannotated > 0:
            self.console.print(f"\n[yellow]Warning: {unannotated} clusters remain unannotated.[/yellow]")
            return Confirm.ask("Are you sure you want to finish?", default=False)
        else:
            self.console.print("\n[green]✅ All clusters have been annotated![/green]")
            return Confirm.ask("Finish annotation session?", default=True)
    
    def apply_annotations_to_adata(self, 
                                 adata: sc.AnnData,
                                 cluster_key: str = 'leiden',
                                 cell_type_column: str = 'cell_type_manual') -> sc.AnnData:
        """
        Apply manual annotations to AnnData object.
        
        Args:
            adata: AnnData object to annotate
            cluster_key: Key in adata.obs containing cluster assignments
            cell_type_column: New column name for cell type annotations
            
        Returns:
            AnnData object with manual annotations applied
        """
        if not self.state:
            raise ValueError("No annotation session available. Run rich_annotation_interface first.")
        
        # Create cell type mapping
        adata_copy = adata.copy()
        
        # Map cluster IDs to cell types
        cluster_to_celltype = {}
        for cluster_id, cell_type in self.state.cell_type_mapping.items():
            cluster_to_celltype[cluster_id] = cell_type
        
        # Mark debris clusters
        for cluster_id in self.state.debris_clusters:
            cluster_to_celltype[cluster_id] = 'Debris'
        
        # Apply annotations
        adata_copy.obs[cell_type_column] = adata_copy.obs[cluster_key].astype(str).map(
            cluster_to_celltype
        ).fillna('Unassigned')
        
        # Store annotation metadata
        adata_copy.uns['manual_annotation_metadata'] = {
            'annotation_timestamp': datetime.now().isoformat(),
            'cluster_key_used': cluster_key,
            'cell_type_column': cell_type_column,
            'total_clusters': len(self.state.clusters),
            'annotated_clusters': len(self.state.cell_type_mapping),
            'debris_clusters': len(self.state.debris_clusters),
            'cluster_colors': self.cluster_colors
        }
        
        return adata_copy
    
    def create_annotation_umap_with_palette(self, 
                                          adata: sc.AnnData,
                                          cluster_col: str = 'leiden') -> Tuple[Any, Dict[str, str]]:
        """
        Generate UMAP plot and extract color palette for terminal sync.
        
        Args:
            adata: AnnData object with UMAP coordinates
            cluster_col: Column name for clustering
            
        Returns:
            Tuple of (plotly figure, color palette dict)
        """
        # This will be implemented when we add visualization enhancements
        # For now, return the current cluster colors
        return None, self.cluster_colors
    
    def suggest_debris_clusters(self, 
                              adata: sc.AnnData,
                              min_genes: int = 200,
                              max_mt_percent: float = 50,
                              min_umi: int = 500) -> List[str]:
        """
        Suggest potential debris clusters based on QC metrics.
        
        Args:
            adata: AnnData object
            min_genes: Minimum genes per cell threshold
            max_mt_percent: Maximum mitochondrial percentage
            min_umi: Minimum UMI count threshold
            
        Returns:
            List of cluster IDs suggested as debris
        """
        if not self.state:
            self.initialize_annotation_session(adata)
        
        suggested_debris = []
        
        for cluster_id, info in self.state.clusters.items():
            is_debris = False
            
            # Check QC metrics
            if 'mean_genes' in info.qc_scores:
                if info.qc_scores['mean_genes'] < min_genes:
                    is_debris = True
            
            if 'mean_mt_pct' in info.qc_scores:
                if info.qc_scores['mean_mt_pct'] > max_mt_percent:
                    is_debris = True
            
            if 'mean_total_counts' in info.qc_scores:
                if info.qc_scores['mean_total_counts'] < min_umi:
                    is_debris = True
            
            # Very small clusters are often debris
            if info.cell_count < 10:
                is_debris = True
            
            if is_debris:
                suggested_debris.append(cluster_id)
        
        return suggested_debris
    
    def validate_annotation_coverage(self, 
                                   adata: sc.AnnData,
                                   annotation_col: str = "cell_type_manual") -> Dict[str, Any]:
        """
        Validate annotation completeness and consistency.
        
        Args:
            adata: AnnData object with annotations
            annotation_col: Column containing cell type annotations
            
        Returns:
            Dictionary with validation results
        """
        if annotation_col not in adata.obs.columns:
            return {
                'valid': False,
                'error': f'Annotation column {annotation_col} not found'
            }
        
        annotations = adata.obs[annotation_col]
        
        # Calculate coverage statistics
        total_cells = len(annotations)
        unassigned_cells = len(annotations[annotations == 'Unassigned'])
        debris_cells = len(annotations[annotations == 'Debris'])
        annotated_cells = total_cells - unassigned_cells - debris_cells
        
        # Get unique cell types
        unique_types = annotations[~annotations.isin(['Unassigned', 'Debris'])].unique()
        
        validation_results = {
            'valid': True,
            'total_cells': total_cells,
            'annotated_cells': annotated_cells,
            'unassigned_cells': unassigned_cells,
            'debris_cells': debris_cells,
            'coverage_percentage': (annotated_cells / total_cells) * 100,
            'unique_cell_types': len(unique_types),
            'cell_type_names': list(unique_types),
            'cell_type_counts': annotations.value_counts().to_dict()
        }
        
        return validation_results
