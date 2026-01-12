"""
Configuration for research agent.

This module defines field categorization for GEO metadata verbosity control.
"""

__all__ = [
    "ESSENTIAL_FIELDS",
    "STANDARD_FIELDS",
    "VERBOSE_FIELDS",
]

# ============================================================
# GEO Metadata Verbosity Control
# ============================================================
# Field categorization for controlling metadata output verbosity.
# Used by get_dataset_metadata tool to prevent context overflow.

ESSENTIAL_FIELDS = {
    "database",
    "geo_accession",
    "title",
    "status",
    "pubmed_id",
    "summary",
}

STANDARD_FIELDS = {
    "overall_design",
    "type",
    "submission_date",
    "last_update_date",
    "web_link",
    "contributor",
    "contact_name",
    "contact_email",
    "contact_institute",
    "contact_country",
    "platform_id",
    "organism",
    "n_samples",
    "sample_count",
}

VERBOSE_FIELDS = {
    "sample_id",
    "contact_phone",
    "contact_department",
    "contact_address",
    "contact_city",
    "contact_zip/postal_code",
    "supplementary_file",
    "platform_taxid",
    "sample_taxid",
    "relation",
    "samples",
    "platforms",
}
