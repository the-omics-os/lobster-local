"""
Publication Resolver for automatic PMID/DOI to PDF URL resolution.

This module provides intelligent PDF access resolution with a tiered waterfall strategy:
1. PubMed Central (PMC) - Free full text via NCBI E-utilities
2. bioRxiv/medRxiv - Preprint servers with direct PDF access
3. Publisher Direct - When open access flag is set
4. Helpful suggestions when paywalled

This eliminates the #1 user pain point: manually finding PDF URLs.
"""

from typing import Any, Dict, List, Optional

import requests

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PublicationResolutionResult:
    """Result of publication resolution attempt."""

    def __init__(
        self,
        identifier: str,
        pdf_url: Optional[str] = None,
        html_url: Optional[str] = None,
        source: str = "unknown",
        access_type: str = "unknown",
        alternative_urls: Optional[List[str]] = None,
        suggestions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize resolution result.

        Args:
            identifier: Original identifier (PMID/DOI)
            pdf_url: Direct PDF URL if found
            source: Resolution source ("pmc" | "biorxiv" | "medrxiv" | "publisher" | "paywalled")
            access_type: Access type ("open_access" | "paywalled" | "preprint")
            alternative_urls: List of alternative access URLs
            suggestions: Human-readable guidance for accessing paper
            metadata: Additional metadata about the resolution
        """
        self.identifier = identifier
        self.pdf_url = pdf_url
        self.html_url = html_url
        self.source = source
        self.access_type = access_type
        self.alternative_urls = alternative_urls or []
        self.suggestions = suggestions or ""
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "identifier": self.identifier,
            "pdf_url": self.pdf_url,
            "html_url": self.html_url,
            "source": self.source,
            "access_type": self.access_type,
            "alternative_urls": self.alternative_urls,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    def is_accessible(self) -> bool:
        """Check if PDF is accessible."""
        has_url = self.pdf_url is not None or self.html_url is not None
        return has_url and self.access_type != "paywalled"


class PublicationResolver:
    """
    Intelligent publication resolver with tiered waterfall strategy.

    Resolution priority:
    1. PubMed Central (PMC) - Best success rate for biomedical papers
    2. bioRxiv/medRxiv - Preprint servers with open access
    3. Publisher Direct - When open access flag is set
    4. Paywalled - Return helpful suggestions
    """

    def __init__(self, timeout: int = 30, cache_ttl: int = 300):
        """
        Initialize resolver with instance-level caching.

        Args:
            timeout: Request timeout in seconds (default: 30)
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache = {}  # Instance-level cache: {identifier: (result, timestamp)}
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Lobster AI Research Tool/1.0 (mailto:support@omics-os.com)"}
        )
        logger.info("Initialized PublicationResolver with caching")

    def resolve(self, identifier: str) -> PublicationResolutionResult:
        """
        Resolve identifier to PDF URL using tiered waterfall strategy.

        Uses instance-level caching to avoid redundant API calls.

        Args:
            identifier: PMID, DOI, or publication identifier

        Returns:
            PublicationResolutionResult with access information

        Examples:
            >>> resolver = PublicationResolver()
            >>> result = resolver.resolve("PMID:12345678")
            >>> if result.is_accessible():
            ...     print(f"PDF available at: {result.pdf_url}")
            >>> else:
            ...     print(f"Suggestions: {result.suggestions}")
        """
        logger.info(f"Resolving identifier: {identifier}")

        # Normalize identifier
        identifier = identifier.strip()

        # Check cache first
        import time

        if identifier in self._cache:
            cached_result, cached_time = self._cache[identifier]
            age = time.time() - cached_time
            if age < self.cache_ttl:
                logger.debug(
                    f"Cache hit for {identifier} (age: {age:.1f}s, TTL: {self.cache_ttl}s)"
                )
                return cached_result
            else:
                logger.debug(f"Cache expired for {identifier} (age: {age:.1f}s)")
                del self._cache[identifier]

        pmid, doi = self._parse_identifier(identifier)

        # Strategy 1: Try PubMed Central (PMC) first
        if pmid:
            result = self._resolve_via_pmc(pmid)
            if result.is_accessible():
                logger.info(f"Resolved via PMC: {result.pdf_url}")
                self._cache[identifier] = (result, time.time())
                return result

        # Strategy 1.5: Try NCBI LinkOut for publisher URLs
        if pmid:
            result = self._resolve_via_linkout(pmid)
            if result.is_accessible():
                logger.info(f"Resolved via LinkOut: {result.pdf_url}")
                self._cache[identifier] = (result, time.time())
                return result

        # Strategy 1.75: If we have PMID but no DOI, fetch DOI from PubMed
        # This unlocks preprint and publisher resolution strategies
        if pmid and not doi:
            doi = self._get_doi_from_pmid(pmid)
            if doi:
                logger.info(f"Fetched DOI from PubMed: {doi}")

        # Strategy 2: Try bioRxiv/medRxiv preprints
        if doi:
            result = self._resolve_via_preprint_servers(doi)
            if result.is_accessible():
                logger.info(f"Resolved via preprint server: {result.pdf_url}")
                self._cache[identifier] = (result, time.time())
                return result

        # Strategy 3: Try publisher direct (limited support)
        if doi:
            result = self._resolve_via_publisher(doi)
            if result.is_accessible():
                logger.info(f"Resolved via publisher: {result.pdf_url}")
                self._cache[identifier] = (result, time.time())
                return result

        # Strategy 4: Generate helpful suggestions for paywalled papers
        logger.info(f"Paper appears paywalled: {identifier}")
        result = self._generate_access_suggestions(identifier, pmid, doi)
        self._cache[identifier] = (result, time.time())
        return result

    def _parse_identifier(self, identifier: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse identifier to extract PMID and/or DOI.

        Args:
            identifier: Input identifier

        Returns:
            Tuple of (pmid, doi)
        """
        pmid = None
        doi = None

        # Check for PMID
        if identifier.upper().startswith("PMID:"):
            pmid = identifier[5:].strip()
        elif identifier.isdigit() and len(identifier) <= 8:
            pmid = identifier

        # Check for DOI
        if identifier.startswith("10."):
            doi = identifier
        elif "doi.org/" in identifier.lower():
            doi = identifier.split("doi.org/")[-1]

        return pmid, doi

    def _get_doi_from_pmid(self, pmid: str) -> Optional[str]:
        """
        Fetch DOI from PubMed metadata using NCBI EFetch API.

        This method enables the waterfall strategy to continue even when
        only a PMID is provided, unlocking preprint and publisher resolution.

        Args:
            pmid: PubMed ID

        Returns:
            DOI string if found, None otherwise

        Example:
            >>> resolver = PublicationResolver()
            >>> doi = resolver._get_doi_from_pmid("37963457")
            >>> print(doi)  # "10.1016/j.immuni.2023.10.001"
        """
        import xml.etree.ElementTree as ET

        logger.info(f"Fetching DOI from PubMed for PMID: {pmid}")

        try:
            # Use NCBI EFetch to get PubMed record in JSON format
            efetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
            )

            response = self.session.get(efetch_url, timeout=self.timeout)
            response.raise_for_status()

            # Parse XML response for DOI

            root = ET.fromstring(response.content)

            # Look for DOI in ArticleId elements
            # Path: PubmedArticle/PubmedData/ArticleIdList/ArticleId[@IdType="doi"]
            for article_id in root.findall(".//ArticleId[@IdType='doi']"):
                doi = article_id.text
                if doi:
                    logger.info(f"Found DOI for PMID {pmid}: {doi}")
                    return doi.strip()

            logger.debug(f"No DOI found in PubMed record for PMID {pmid}")
            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching DOI for PMID {pmid}: {e}")
            return None
        except ET.ParseError as e:
            logger.warning(f"Error parsing PubMed XML for PMID {pmid}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error fetching DOI for PMID {pmid}: {e}")
            return None

    def _resolve_via_pmc(self, pmid: str) -> PublicationResolutionResult:
        """
        Resolve PMID to PDF via PubMed Central.

        Uses NCBI E-utilities API to check for free full text in PMC.

        Args:
            pmid: PubMed ID

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking PMC for PMID: {pmid}")

        try:
            # Step 1: Use elink to find PMC ID
            elink_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                f"?dbfrom=pubmed&db=pmc&id={pmid}&linkname=pubmed_pmc&retmode=json"
            )

            response = self.session.get(elink_url, timeout=self.timeout)
            response.raise_for_status()

            # Validate JSON response structure
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response from PMC API for PMID {pmid}: {e}")
                return PublicationResolutionResult(
                    identifier=f"PMID:{pmid}", source="pmc", access_type="error"
                )

            # Validate response is a dictionary with expected structure
            if not isinstance(data, dict):
                logger.error(
                    f"PMC API returned non-dict response for PMID {pmid}: {type(data)}"
                )
                return PublicationResolutionResult(
                    identifier=f"PMID:{pmid}", source="pmc", access_type="error"
                )

            # Extract PMC ID if available
            pmc_id = None
            try:
                linksets = data.get("linksets", [])
                if not isinstance(linksets, list):
                    logger.warning(
                        f"PMC API linksets is not a list for PMID {pmid}: {type(linksets)}"
                    )
                elif linksets and len(linksets) > 0:
                    linksetdbs = linksets[0].get("linksetdbs", [])
                    if not isinstance(linksetdbs, list):
                        logger.warning(
                            f"PMC API linksetdbs is not a list for PMID {pmid}: {type(linksetdbs)}"
                        )
                    elif linksetdbs and len(linksetdbs) > 0:
                        links = linksetdbs[0].get("links", [])
                        if not isinstance(links, list):
                            logger.warning(
                                f"PMC API links is not a list for PMID {pmid}: {type(links)}"
                            )
                        elif links and len(links) > 0:
                            pmc_id = links[0]
                            # Validate PMC ID is numeric
                            if not isinstance(pmc_id, (int, str)) or (
                                isinstance(pmc_id, str) and not pmc_id.isdigit()
                            ):
                                logger.warning(
                                    f"PMC API returned invalid PMC ID for PMID {pmid}: {pmc_id}"
                                )
                                pmc_id = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"No PMC link found for PMID {pmid}: {e}")

            if not pmc_id:
                return PublicationResolutionResult(
                    identifier=f"PMID:{pmid}",
                    source="pmc",
                    access_type="not_in_pmc",
                )

            # Step 2: Construct PMC HTML + PDF URLs
            html_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

            logger.info(f"Found PMC article: PMC{pmc_id}")

            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}",
                pdf_url=pdf_url,
                html_url=html_url,
                source="pmc",
                access_type="open_access",
                alternative_urls=[html_url],
                metadata={"pmc_id": f"PMC{pmc_id}"},
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying PMC for PMID {pmid}: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="pmc", access_type="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error in PMC resolution: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="pmc", access_type="error"
            )

    def _resolve_via_linkout(self, pmid: str) -> PublicationResolutionResult:
        """
        Resolve PMID to publisher URL using NCBI LinkOut service.

        LinkOut provides direct publisher URLs with better success rate than
        CrossRef for biomedical papers. This is especially useful for papers
        with institutional access or open access at publisher sites.

        Args:
            pmid: PubMed ID

        Returns:
            PublicationResolutionResult with publisher URL or not_available

        Note:
            This method returns publisher URLs which may be paywalled.
            The waterfall strategy will continue to other methods if needed.
        """
        logger.info(f"Checking NCBI LinkOut for PMID: {pmid}")

        try:
            # Use ELink with prlinks (provider links) to get publisher URLs
            linkout_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                f"?dbfrom=pubmed&id={pmid}&cmd=prlinks&retmode=json"
            )

            response = self.session.get(linkout_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Extract provider URL if available
            # LinkOut prlinks returns: {linksets: [{idurllist: [{objurls: [{url: ...}]}]}]}
            try:
                linksets = data.get("linksets", [])
                if linksets and len(linksets) > 0:
                    idurllist = linksets[0].get("idurllist", [])
                    if idurllist and len(idurllist) > 0:
                        objurls = idurllist[0].get("objurls", [])
                        if objurls and len(objurls) > 0:
                            url_data = objurls[0].get("url", {})
                            provider_url = url_data.get("value")

                            if provider_url:
                                logger.info(
                                    f"Found LinkOut URL for PMID {pmid}: {provider_url}"
                                )
                                provider_url_lower = provider_url.lower()
                                url_kwargs = (
                                    {"pdf_url": provider_url}
                                    if provider_url_lower.endswith(".pdf")
                                    else {"html_url": provider_url}
                                )

                                return PublicationResolutionResult(
                                    identifier=f"PMID:{pmid}",
                                    source="linkout",
                                    access_type="publisher",
                                    metadata={
                                        "linkout_provider": url_data.get(
                                            "provider", {}
                                        ).get("name", "Unknown")
                                    },
                                    **url_kwargs,
                                )
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"No LinkOut URL found for PMID {pmid}: {e}")

            # No LinkOut URL found
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}",
                source="linkout",
                access_type="not_available",
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying LinkOut for PMID {pmid}: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="linkout", access_type="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error in LinkOut resolution: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="linkout", access_type="error"
            )

    def _resolve_via_preprint_servers(self, doi: str) -> PublicationResolutionResult:
        """
        Resolve DOI to PDF via bioRxiv/medRxiv preprint servers.

        Args:
            doi: Digital Object Identifier

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking preprint servers for DOI: {doi}")

        # Check if DOI is from bioRxiv or medRxiv
        if "biorxiv.org" in doi.lower() or doi.startswith("10.1101/"):
            content_base = f"https://www.biorxiv.org/content/{doi}"
            html_url = f"{content_base}.full"
            pdf_url = f"{html_url}.pdf"

            return PublicationResolutionResult(
                identifier=doi,
                pdf_url=pdf_url,
                html_url=html_url,
                source="biorxiv",
                access_type="preprint",
                alternative_urls=[content_base],
                metadata={"server": "biorxiv"},
            )

        elif "medrxiv.org" in doi.lower():
            content_base = f"https://www.medrxiv.org/content/{doi}"
            html_url = f"{content_base}.full"
            pdf_url = f"{html_url}.pdf"

            return PublicationResolutionResult(
                identifier=doi,
                pdf_url=pdf_url,
                html_url=html_url,
                source="medrxiv",
                access_type="preprint",
                alternative_urls=[content_base],
                metadata={"server": "medrxiv"},
            )

        # Not a preprint server DOI
        return PublicationResolutionResult(
            identifier=doi, source="preprint", access_type="not_preprint"
        )

    def _resolve_via_publisher(self, doi: str) -> PublicationResolutionResult:
        """
        Resolve DOI to PDF via publisher (limited support for open access).

        This is a fallback strategy with limited success rate.

        Args:
            doi: Digital Object Identifier

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking publisher for DOI: {doi}")

        try:
            # Use CrossRef API to get metadata
            crossref_url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(crossref_url, timeout=self.timeout)
            response.raise_for_status()

            # Validate JSON response structure
            try:
                data = response.json()
            except ValueError as e:
                logger.error(
                    f"Invalid JSON response from CrossRef API for DOI {doi}: {e}"
                )
                return PublicationResolutionResult(
                    identifier=doi, source="publisher", access_type="error"
                )

            # Validate response is a dictionary with expected structure
            if not isinstance(data, dict):
                logger.error(
                    f"CrossRef API returned non-dict response for DOI {doi}: {type(data)}"
                )
                return PublicationResolutionResult(
                    identifier=doi, source="publisher", access_type="error"
                )

            message = data.get("message", {})
            if not isinstance(message, dict):
                logger.warning(
                    f"CrossRef API message is not a dict for DOI {doi}: {type(message)}"
                )
                message = {}

            is_open_access = False

            # Check for open access indicators
            license_info = message.get("license", [])
            if not isinstance(license_info, list):
                logger.warning(
                    f"CrossRef API license info is not a list for DOI {doi}: {type(license_info)}"
                )
                license_info = []

            for license_item in license_info:
                if (
                    isinstance(license_item, dict)
                    and "open-access" in str(license_item).lower()
                ):
                    is_open_access = True
                    break

            if not is_open_access:
                link = message.get("link", [])
                if not isinstance(link, list):
                    logger.warning(
                        f"CrossRef API link is not a list for DOI {doi}: {type(link)}"
                    )
                    link = []

                for link_item in link:
                    if not isinstance(link_item, dict):
                        continue
                    if link_item.get(
                        "content-type"
                    ) == "application/pdf" and "unixref.org" not in link_item.get(
                        "URL", ""
                    ):
                        # Found a direct PDF link
                        pdf_url = link_item.get("URL")
                        if pdf_url:  # Validate URL exists
                            return PublicationResolutionResult(
                                identifier=doi,
                                pdf_url=pdf_url,
                                source="publisher",
                                access_type="open_access",
                                metadata={"publisher": message.get("publisher")},
                            )

            # No direct PDF found
            return PublicationResolutionResult(
                identifier=doi,
                source="publisher",
                access_type="not_open_access",
                metadata={"publisher": message.get("publisher")},
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying CrossRef for DOI {doi}: {e}")
            return PublicationResolutionResult(
                identifier=doi, source="publisher", access_type="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error in publisher resolution: {e}")
            return PublicationResolutionResult(
                identifier=doi, source="publisher", access_type="error"
            )

    def _generate_access_suggestions(
        self, identifier: str, pmid: Optional[str], doi: Optional[str]
    ) -> PublicationResolutionResult:
        """
        Generate helpful suggestions when paper is paywalled.

        Args:
            identifier: Original identifier
            pmid: PMID if available
            doi: DOI if available

        Returns:
            PublicationResolutionResult with suggestions
        """
        suggestions = []
        alternative_urls = []

        suggestions.append("## Alternative Access Options\n")

        # Suggestion 1: PMC Accepted Manuscript
        if pmid:
            suggestions.append(
                f"1. **PubMed Central Accepted Manuscript**: Check if an accepted manuscript version is available:\n"
                f"   - https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}\n"
            )
            alternative_urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}")

        # Suggestion 2: bioRxiv/medRxiv search
        if doi or pmid:
            search_term = doi if doi else pmid
            suggestions.append(
                f"2. **Preprint Servers**: Check for preprints on bioRxiv or medRxiv:\n"
                f"   - bioRxiv: https://www.biorxiv.org/search/{search_term}\n"
                f"   - medRxiv: https://www.medrxiv.org/search/{search_term}\n"
            )
            alternative_urls.append(f"https://www.biorxiv.org/search/{search_term}")

        # Suggestion 3: Institutional Access
        suggestions.append(
            "3. **Institutional Access**: If you're affiliated with a university, try:\n"
            "   - Accessing through your institution's library proxy\n"
            "   - Using VPN to connect to institutional network\n"
            "   - Requesting through interlibrary loan\n"
        )

        # Suggestion 4: Author Contact
        if pmid:
            suggestions.append(
                f"4. **Contact Authors**: You can:\n"
                f"   - Email the corresponding author to request a PDF\n"
                f"   - Check author profiles on ResearchGate or Academia.edu\n"
                f"   - PubMed author info: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )
            alternative_urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

        # Suggestion 5: Unpaywall
        if doi:
            suggestions.append(
                f"5. **Unpaywall Service**: Check for legal open access versions:\n"
                f"   - https://unpaywall.org/{doi}\n"
            )
            alternative_urls.append(f"https://unpaywall.org/{doi}")

        suggestions.append(
            "\n💡 **Tip**: Many publishers allow authors to share accepted manuscripts. "
            "Contacting the corresponding author is often successful!"
        )

        return PublicationResolutionResult(
            identifier=identifier,
            pdf_url=None,
            source="paywalled",
            access_type="paywalled",
            alternative_urls=alternative_urls,
            suggestions="\n".join(suggestions),
            metadata={"pmid": pmid, "doi": doi},
        )

    def batch_resolve(
        self, identifiers: List[str], max_batch: int = 5
    ) -> List[PublicationResolutionResult]:
        """
        Resolve multiple identifiers sequentially (conservative approach).

        Args:
            identifiers: List of PMIDs/DOIs to resolve
            max_batch: Maximum batch size (default: 5)

        Returns:
            List of PublicationResolutionResult objects

        Examples:
            >>> resolver = PublicationResolver()
            >>> identifiers = ["PMID:12345678", "10.1038/s41586-021-12345-6"]
            >>> results = resolver.batch_resolve(identifiers)
            >>> for result in results:
            ...     if result.is_accessible():
            ...         print(f"✅ {result.identifier}: {result.pdf_url}")
            ...     else:
            ...         print(f"❌ {result.identifier}: Paywalled")
        """
        logger.info(f"Batch resolving {len(identifiers)} identifiers")

        # Limit batch size
        if len(identifiers) > max_batch:
            logger.warning(
                f"Batch size {len(identifiers)} exceeds max {max_batch}, truncating"
            )
            identifiers = identifiers[:max_batch]

        results = []
        for i, identifier in enumerate(identifiers, 1):
            logger.info(f"Processing {i}/{len(identifiers)}: {identifier}")
            try:
                result = self.resolve(identifier)
                results.append(result)
            except Exception as e:
                logger.error(f"Error resolving {identifier}: {e}")
                results.append(
                    PublicationResolutionResult(
                        identifier=identifier,
                        source="error",
                        access_type="error",
                        suggestions=f"Error during resolution: {str(e)}",
                    )
                )

        logger.info(
            f"Batch resolution complete: {sum(1 for r in results if r.is_accessible())}/{len(results)} accessible"
        )
        return results
