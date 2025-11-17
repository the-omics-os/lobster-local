"""
IR Coverage Reporter for Service-Emitted Intermediate Representation.

This module provides functionality to analyze and report on IR coverage across
the Lobster codebase, tracking which services emit IR and which tools are covered.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ServiceCoverage:
    """
    Coverage information for a single service.

    Attributes:
        service_name: Name of the service module
        service_path: Path to the service file
        has_ir_import: Whether the service imports AnalysisStep
        methods_with_ir: Set of method names that emit IR
        methods_without_ir: Set of method names without IR
        total_methods: Total number of public methods
        coverage_percentage: Percentage of methods with IR
    """

    service_name: str
    service_path: Path
    has_ir_import: bool = False
    methods_with_ir: Set[str] = field(default_factory=set)
    methods_without_ir: Set[str] = field(default_factory=set)
    total_methods: int = 0
    coverage_percentage: float = 0.0

    def __post_init__(self):
        """Calculate coverage percentage after initialization."""
        self.total_methods = len(self.methods_with_ir) + len(self.methods_without_ir)
        if self.total_methods > 0:
            self.coverage_percentage = (
                len(self.methods_with_ir) / self.total_methods
            ) * 100


@dataclass
class CoverageReport:
    """
    Comprehensive coverage report for all services.

    Attributes:
        services: Dictionary mapping service names to ServiceCoverage objects
        total_services: Total number of services analyzed
        services_with_ir: Number of services emitting IR
        total_methods: Total number of methods across all services
        methods_with_ir: Total number of methods emitting IR
        overall_coverage: Overall IR coverage percentage
        timestamp: Report generation timestamp
    """

    services: Dict[str, ServiceCoverage] = field(default_factory=dict)
    total_services: int = 0
    services_with_ir: int = 0
    total_methods: int = 0
    methods_with_ir: int = 0
    overall_coverage: float = 0.0
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Calculate aggregate statistics after initialization."""
        self.total_services = len(self.services)
        self.services_with_ir = sum(
            1 for svc in self.services.values() if svc.has_ir_import
        )
        self.total_methods = sum(svc.total_methods for svc in self.services.values())
        self.methods_with_ir = sum(
            len(svc.methods_with_ir) for svc in self.services.values()
        )

        if self.total_methods > 0:
            self.overall_coverage = (self.methods_with_ir / self.total_methods) * 100

    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_services": self.total_services,
                "services_with_ir": self.services_with_ir,
                "total_methods": self.total_methods,
                "methods_with_ir": self.methods_with_ir,
                "overall_coverage": round(self.overall_coverage, 2),
            },
            "services": {
                name: {
                    "path": str(svc.service_path),
                    "has_ir_import": svc.has_ir_import,
                    "methods_with_ir": list(svc.methods_with_ir),
                    "methods_without_ir": list(svc.methods_without_ir),
                    "total_methods": svc.total_methods,
                    "coverage_percentage": round(svc.coverage_percentage, 2),
                }
                for name, svc in self.services.items()
            },
        }


class IRCoverageAnalyzer:
    """
    Analyzer for IR coverage across service files.

    Scans service modules to identify:
    - Which services import AnalysisStep
    - Which methods return IR (3-tuple pattern)
    - Coverage percentage per service and overall
    """

    def __init__(self, services_dir: Optional[Path] = None):
        """
        Initialize coverage analyzer.

        Args:
            services_dir: Path to services directory (defaults to lobster/tools/)
        """
        if services_dir is None:
            # Default to lobster/tools/ directory
            self.services_dir = Path(__file__).parent.parent / "tools"
        else:
            self.services_dir = Path(services_dir)

        logger.debug(
            f"Initialized IRCoverageAnalyzer with services_dir: {self.services_dir}"
        )

    def analyze_service_file(self, service_path: Path) -> ServiceCoverage:
        """
        Analyze a single service file for IR coverage.

        Args:
            service_path: Path to service file

        Returns:
            ServiceCoverage object with analysis results
        """
        service_name = service_path.stem
        coverage = ServiceCoverage(
            service_name=service_name,
            service_path=service_path,
        )

        try:
            with open(service_path, "r") as f:
                source = f.read()

            tree = ast.parse(source)

            # Check for AnalysisStep import
            coverage.has_ir_import = self._has_analysis_step_import(tree)

            # Analyze methods for IR emission
            methods_with_ir, methods_without_ir = self._analyze_methods(tree)

            coverage.methods_with_ir = methods_with_ir
            coverage.methods_without_ir = methods_without_ir

            # Recalculate coverage after setting methods
            coverage.__post_init__()

        except Exception as e:
            logger.error(f"Error analyzing {service_path}: {e}")

        return coverage

    def _has_analysis_step_import(self, tree: ast.AST) -> bool:
        """
        Check if the module imports AnalysisStep.

        Args:
            tree: Parsed AST tree

        Returns:
            True if AnalysisStep is imported
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "analysis_ir" in node.module:
                    for alias in node.names:
                        if alias.name == "AnalysisStep":
                            return True

        return False

    def _analyze_methods(self, tree: ast.AST) -> Tuple[Set[str], Set[str]]:
        """
        Analyze methods in the module for IR emission.

        A method is considered to emit IR if:
        1. It has a return statement with a tuple
        2. The tuple has 3 elements
        3. The last element looks like an AnalysisStep (variable name contains 'ir' or 'step')

        Args:
            tree: Parsed AST tree

        Returns:
            Tuple of (methods_with_ir, methods_without_ir)
        """
        methods_with_ir = set()
        methods_without_ir = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Analyze methods in classes (service classes)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name

                        # Skip private methods and special methods
                        if method_name.startswith("_"):
                            continue

                        # Check if method returns 3-tuple with IR
                        if self._method_returns_ir(item):
                            methods_with_ir.add(method_name)
                        else:
                            methods_without_ir.add(method_name)

            elif isinstance(node, ast.FunctionDef):
                # Analyze top-level functions
                func_name = node.name

                # Skip private functions
                if func_name.startswith("_"):
                    continue

                if self._method_returns_ir(node):
                    methods_with_ir.add(func_name)
                else:
                    methods_without_ir.add(func_name)

        return methods_with_ir, methods_without_ir

    def _method_returns_ir(self, func_node: ast.FunctionDef) -> bool:
        """
        Check if a function/method returns a 3-tuple with IR.

        Args:
            func_node: Function definition AST node

        Returns:
            True if function returns 3-tuple with IR pattern
        """
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value and isinstance(node.value, ast.Tuple):
                    # Check if tuple has 3 elements
                    if len(node.value.elts) == 3:
                        # Check if third element looks like IR
                        third_elem = node.value.elts[2]
                        if isinstance(third_elem, ast.Name):
                            var_name = third_elem.id.lower()
                            if (
                                "ir" in var_name
                                or "step" in var_name
                                or "analysis" in var_name
                            ):
                                return True

        return False

    def scan_services(
        self, pattern: str = "*_service.py"
    ) -> Dict[str, ServiceCoverage]:
        """
        Scan all service files matching the pattern.

        Args:
            pattern: Glob pattern for service files (default: *_service.py)

        Returns:
            Dictionary mapping service names to ServiceCoverage objects
        """
        services = {}

        if not self.services_dir.exists():
            logger.warning(f"Services directory not found: {self.services_dir}")
            return services

        for service_path in self.services_dir.glob(pattern):
            if service_path.is_file():
                coverage = self.analyze_service_file(service_path)
                services[coverage.service_name] = coverage

        return services

    def generate_report(self, pattern: str = "*_service.py") -> CoverageReport:
        """
        Generate comprehensive coverage report for all services.

        Args:
            pattern: Glob pattern for service files

        Returns:
            CoverageReport with full analysis results
        """
        from datetime import datetime

        services = self.scan_services(pattern)

        report = CoverageReport(
            services=services,
            timestamp=datetime.now().isoformat(),
        )

        # Recalculate aggregate statistics
        report.__post_init__()

        return report

    def print_report(self, report: CoverageReport, verbose: bool = False) -> None:
        """
        Print coverage report to console.

        Args:
            report: CoverageReport to display
            verbose: Whether to show detailed method lists
        """
        print("\n" + "=" * 80)
        print("IR COVERAGE REPORT")
        print("=" * 80)
        print(f"Generated: {report.timestamp}")
        print()

        # Summary section
        print("SUMMARY:")
        print(f"  Total Services: {report.total_services}")
        print(f"  Services with IR: {report.services_with_ir}")
        print(f"  Total Methods: {report.total_methods}")
        print(f"  Methods with IR: {report.methods_with_ir}")
        print(f"  Overall Coverage: {report.overall_coverage:.2f}%")
        print()

        # Per-service breakdown
        print("SERVICE BREAKDOWN:")
        print("-" * 80)

        # Sort services by coverage percentage (descending)
        sorted_services = sorted(
            report.services.items(),
            key=lambda x: x[1].coverage_percentage,
            reverse=True,
        )

        for service_name, coverage in sorted_services:
            status_icon = "✓" if coverage.has_ir_import else "✗"
            print(f"{status_icon} {service_name}")
            print(f"   Coverage: {coverage.coverage_percentage:.2f}%")
            print(
                f"   Methods: {len(coverage.methods_with_ir)}/{coverage.total_methods} with IR"
            )

            if verbose:
                if coverage.methods_with_ir:
                    print(f"   With IR: {', '.join(sorted(coverage.methods_with_ir))}")
                if coverage.methods_without_ir:
                    print(
                        f"   Without IR: {', '.join(sorted(coverage.methods_without_ir))}"
                    )

            print()

        print("=" * 80)

    def identify_gaps(self, report: CoverageReport) -> List[Tuple[str, str]]:
        """
        Identify services and methods without IR coverage.

        Args:
            report: CoverageReport to analyze

        Returns:
            List of (service_name, method_name) tuples for methods without IR
        """
        gaps = []

        for service_name, coverage in report.services.items():
            for method_name in coverage.methods_without_ir:
                gaps.append((service_name, method_name))

        return gaps


def main():
    """CLI entry point for IR coverage analysis."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Analyze IR coverage in Lobster services"
    )
    parser.add_argument(
        "--services-dir",
        type=Path,
        help="Path to services directory (default: lobster/tools/)",
    )
    parser.add_argument(
        "--pattern",
        default="*_service.py",
        help="Glob pattern for service files (default: *_service.py)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed method lists",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Export report as JSON to specified file",
    )
    parser.add_argument(
        "--gaps-only",
        action="store_true",
        help="Only show methods without IR coverage",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = IRCoverageAnalyzer(services_dir=args.services_dir)

    # Generate report
    report = analyzer.generate_report(pattern=args.pattern)

    # Print report
    if args.gaps_only:
        gaps = analyzer.identify_gaps(report)
        print("\nMETHODS WITHOUT IR COVERAGE:")
        print("-" * 80)
        for service_name, method_name in sorted(gaps):
            print(f"{service_name}.{method_name}")
        print(f"\nTotal gaps: {len(gaps)}")
    else:
        analyzer.print_report(report, verbose=args.verbose)

    # Export JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport exported to: {args.json}")

    # Exit with error code if coverage is below target
    target_coverage = 60.0
    if report.overall_coverage < target_coverage:
        print(
            f"\n⚠️  Coverage ({report.overall_coverage:.2f}%) is below target ({target_coverage}%)"
        )
        sys.exit(1)
    else:
        print(
            f"\n✓ Coverage target met ({report.overall_coverage:.2f}% >= {target_coverage}%)"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
