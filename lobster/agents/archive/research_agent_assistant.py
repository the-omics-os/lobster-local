"""
Research Agent Assistant for metadata validation, PDF resolution, and LLM-based operations.

This module handles:
1. Metadata validation for datasets (pre-download validation)
2. PDF URL resolution from PMIDs/DOIs (automatic access discovery)
3. LLM-based intelligent decision making

This eliminates manual PDF discovery - the #1 user pain point.
"""

from typing import Dict, List, Optional

from langchain_aws import ChatBedrockConverse

from lobster.config.settings import get_settings
from lobster.tools.providers.publication_resolver import (
    PublicationResolutionResult,
    PublicationResolver,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchAgentAssistant:
    """
    Assistant class for PDF resolution and publication access.

    Phase 2 Note: Metadata validation methods have been extracted to
    MetadataValidationService following the Single Responsibility Principle.
    This class now focuses exclusively on PDF resolution.
    """

    def __init__(self):
        """Initialize the Research Agent Assistant for PDF resolution."""
        self.settings = get_settings()
        self._llm = None
        self._resolver = None

    @property
    def llm(self):
        """Lazy initialization of LLM for intelligent resolution decisions."""
        if self._llm is None:
            llm_params = self.settings.get_agent_llm_params("assistant")
            self._llm = ChatBedrockConverse(**llm_params)
        return self._llm

    @property
    def resolver(self):
        """Lazy initialization of PDF resolver."""
        if self._resolver is None:
            self._resolver = PublicationResolver()
        return self._resolver

    # =====================================
    # PDF Resolution Methods
    # =====================================

    def resolve_publication_to_pdf(
        self, identifier: str
    ) -> PublicationResolutionResult:
        """
        Resolve PMID/DOI/publication identifier to accessible PDF URL.

        This is the main entry point for PDF resolution. Uses tiered waterfall strategy:
        1. PubMed Central (PMC) - Free full text
        2. bioRxiv/medRxiv - Preprint servers
        3. Publisher Direct - Open access
        4. Generate suggestions if paywalled

        Args:
            identifier: PMID, DOI, or publication identifier

        Returns:
            PublicationResolutionResult with PDF URL or access suggestions

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> result = assistant.resolve_publication_to_pdf("PMID:12345678")
            >>> if result.is_accessible():
            ...     print(f"PDF: {result.pdf_url}")
            ... else:
            ...     print(f"Suggestions:\n{result.suggestions}")
        """
        logger.info(f"Resolving publication to PDF: {identifier}")

        try:
            result = self.resolver.resolve(identifier)

            # Log resolution outcome
            if result.is_accessible():
                logger.info(
                    f"Successfully resolved {identifier} via {result.source}: {result.pdf_url}"
                )
            else:
                logger.info(
                    f"Paper {identifier} is paywalled or inaccessible, generated suggestions"
                )

            return result

        except Exception as e:
            logger.error(f"Error resolving publication {identifier}: {e}")
            return PublicationResolutionResult(
                identifier=identifier,
                source="error",
                access_type="error",
                suggestions=f"Error during resolution: {str(e)}",
            )

    def resolve_pmid_to_pmc_url(self, pmid: str) -> Optional[str]:
        """
        Check if PMID has free full text in PMC.

        This is a specialized method for PMC-only resolution.

        Args:
            pmid: PubMed ID

        Returns:
            PMC PDF URL if available, None otherwise

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> pdf_url = assistant.resolve_pmid_to_pmc_url("12345678")
            >>> if pdf_url:
            ...     print(f"PMC PDF available: {pdf_url}")
        """
        logger.info(f"Checking PMC for PMID: {pmid}")

        try:
            result = self.resolver._resolve_via_pmc(pmid)

            if result.is_accessible():
                return result.pdf_url
            else:
                return None

        except Exception as e:
            logger.error(f"Error checking PMC for PMID {pmid}: {e}")
            return None

    def resolve_doi_to_preprint(self, doi: str) -> Optional[str]:
        """
        Check if DOI corresponds to bioRxiv/medRxiv preprint.

        Args:
            doi: Digital Object Identifier

        Returns:
            Preprint PDF URL if available, None otherwise

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> pdf_url = assistant.resolve_doi_to_preprint("10.1101/2024.01.001")
            >>> if pdf_url:
            ...     print(f"Preprint available: {pdf_url}")
        """
        logger.info(f"Checking preprint servers for DOI: {doi}")

        try:
            result = self.resolver._resolve_via_preprint_servers(doi)

            if result.is_accessible():
                return result.pdf_url
            else:
                return None

        except Exception as e:
            logger.error(f"Error checking preprint servers for DOI {doi}: {e}")
            return None

    def generate_access_suggestions(
        self, identifier: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate helpful suggestions when PDF unavailable.

        Args:
            identifier: Publication identifier (PMID/DOI)
            metadata: Optional publication metadata

        Returns:
            Formatted markdown with access suggestions

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> suggestions = assistant.generate_access_suggestions("PMID:12345678")
            >>> print(suggestions)
        """
        logger.info(f"Generating access suggestions for {identifier}")

        # Extract PMID and DOI if available
        pmid = None
        doi = None

        if identifier.upper().startswith("PMID:"):
            pmid = identifier[5:].strip()
        elif identifier.isdigit():
            pmid = identifier

        if identifier.startswith("10."):
            doi = identifier

        # If metadata provided, extract identifiers
        if metadata:
            if not pmid and "pmid" in metadata:
                pmid = metadata["pmid"]
            if not doi and "doi" in metadata:
                doi = metadata["doi"]

        # Use resolver to generate suggestions
        result = self.resolver._generate_access_suggestions(identifier, pmid, doi)

        return result.suggestions

    def batch_resolve_publications(
        self, identifiers: List[str], max_batch: int = 5
    ) -> List[PublicationResolutionResult]:
        """
        Resolve multiple publications sequentially (conservative approach).

        Args:
            identifiers: List of PMIDs/DOIs to resolve
            max_batch: Maximum batch size (default: 5)

        Returns:
            List of PublicationResolutionResult objects

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> identifiers = ["PMID:12345678", "10.1038/s41586-021-12345-6"]
            >>> results = assistant.batch_resolve_publications(identifiers)
            >>> accessible = [r for r in results if r.is_accessible()]
            >>> print(f"Accessible: {len(accessible)}/{len(results)}")
        """
        logger.info(f"Batch resolving {len(identifiers)} publications")

        try:
            results = self.resolver.batch_resolve(identifiers, max_batch=max_batch)

            # Log summary
            accessible_count = sum(1 for r in results if r.is_accessible())
            logger.info(
                f"Batch resolution complete: {accessible_count}/{len(results)} accessible"
            )

            return results

        except Exception as e:
            logger.error(f"Error in batch resolution: {e}")
            # Return error results for all identifiers
            return [
                PublicationResolutionResult(
                    identifier=id_,
                    source="error",
                    access_type="error",
                    suggestions=f"Batch resolution error: {str(e)}",
                )
                for id_ in identifiers
            ]

    def format_resolution_report(
        self, result: PublicationResolutionResult, include_alternatives: bool = True
    ) -> str:
        """
        Format resolution result into human-readable report.

        Args:
            result: PublicationResolutionResult to format
            include_alternatives: Whether to include alternative access options

        Returns:
            Formatted markdown report

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> result = assistant.resolve_publication_to_pdf("PMID:12345678")
            >>> report = assistant.format_resolution_report(result)
            >>> print(report)
        """
        if result.is_accessible():
            # Success report
            status_emoji = "✅"
            report = f"""
## PDF Access Report for {result.identifier}

**Status:** {status_emoji} **ACCESSIBLE**
**Source:** {result.source.upper()}
**Access Type:** {result.access_type}
**PDF URL:** {result.pdf_url}
"""

            if result.alternative_urls and include_alternatives:
                report += "\n### Alternative Access Options:\n"
                for url in result.alternative_urls:
                    report += f"- {url}\n"

        else:
            # Paywalled or inaccessible report
            status_emoji = "❌" if result.access_type == "paywalled" else "⚠️"
            report = f"""
## PDF Access Report for {result.identifier}

**Status:** {status_emoji} **NOT DIRECTLY ACCESSIBLE**
**Access Type:** {result.access_type}

{result.suggestions}
"""

        return report

    def format_batch_resolution_report(
        self, results: List[PublicationResolutionResult]
    ) -> str:
        """
        Format batch resolution results into comprehensive report.

        Args:
            results: List of PublicationResolutionResult objects

        Returns:
            Formatted markdown report with summary and details

        Examples:
            >>> assistant = ResearchAgentAssistant()
            >>> identifiers = ["PMID:12345678", "PMID:87654321"]
            >>> results = assistant.batch_resolve_publications(identifiers)
            >>> report = assistant.format_batch_resolution_report(results)
            >>> print(report)
        """
        accessible = [r for r in results if r.is_accessible()]
        paywalled = [
            r for r in results if not r.is_accessible() and r.access_type == "paywalled"
        ]
        errors = [r for r in results if r.access_type == "error"]

        report = f"""
## Batch PDF Resolution Report

**Total Papers:** {len(results)}
**Accessible:** ✅ {len(accessible)} ({len(accessible)*100//len(results) if results else 0}%)
**Paywalled:** ❌ {len(paywalled)} ({len(paywalled)*100//len(results) if results else 0}%)
**Errors:** ⚠️ {len(errors)} ({len(errors)*100//len(results) if results else 0}%)

---

### ✅ Accessible Papers ({len(accessible)}):
"""

        for r in accessible:
            report += f"\n**{r.identifier}**\n"
            report += f"- Source: {r.source}\n"
            report += f"- PDF URL: {r.pdf_url}\n"

        if paywalled:
            report += f"\n\n### ❌ Paywalled Papers ({len(paywalled)}):\n"
            for r in paywalled:
                report += f"\n**{r.identifier}**\n"
                report += "- See alternative access options below\n"

        if errors:
            report += f"\n\n### ⚠️ Errors ({len(errors)}):\n"
            for r in errors:
                report += f"\n**{r.identifier}**\n"
                report += f"- Error: {r.suggestions}\n"

        # Add detailed suggestions for paywalled papers
        if paywalled:
            report += (
                "\n\n---\n\n## Alternative Access Strategies for Paywalled Papers\n"
            )
            for r in paywalled:
                report += f"\n### {r.identifier}\n"
                report += f"{r.suggestions}\n"

        return report
