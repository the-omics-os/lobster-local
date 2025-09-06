1. Data Ingestion & Provenance
	•	Raw data formats supported (e.g., FASTQ/BAM for transcriptomics, mzML/vendor RAW for proteomics).
	•	Metadata schema present (sample prep, platform, instrument, run date, identifiers).
	•	File integrity checks enabled (checksums, format validation).
	•	Provenance logging system implemented (tool versions, parameters, timestamps).

2. Sample-Level QC
	•	Automated QC pipeline implemented for raw data (basic read/scan/feature quality).
	•	Summary reports generated per sample and aggregated across dataset.
	•	Outlier detection procedures defined (flagging unexpected patterns or sample swaps).

3. Normalization & Scaling
	•	Normalization strategy defined for transcriptomics and proteomics.
	•	Original/raw values preserved alongside normalized data.
	•	Transformations documented (e.g., log, scaling, or normalization method applied).

4. Batch Effects & Technical Variability
	•	Batch detection checks implemented (dimensionality reduction, clustering, covariate review).
	•	Batch correction strategy available (applied only with justification, pre/post documented).
	•	Technical covariates captured in metadata.

5. Missing Data & Imputation
	•	Missingness profiles assessed (extent, random vs systematic).
	•	Imputation strategy defined (appropriate for data type).
	•	Imputation decisions documented in provenance.

6. Identifier Mapping & Harmonization
	•	Standard reference versions defined (e.g., Ensembl, UniProt).
	•	Mapping tables/versioning stored in repository.
	•	Controlled vocabularies/ontologies in use for metadata fields.

7. Filtering & Feature Selection
	•	Low-quality feature filtering rules defined (low counts, low detection).
	•	Filtering thresholds documented and reproducible.
	•	Impact of filtering summarized in QC report.

8. Statistical Confidence & FDR Control
	•	False discovery rate (FDR) thresholds defined for analyses.
	•	Confidence metrics included (p-values, q-values, scores).
	•	Multi-level QC for proteomics (PSM, peptide, protein).

9. Reporting & Documentation
	•	QC dashboards/reports generated (per dataset + global).
	•	Provenance and logs archived with results.
	•	Reproducibility ensured (containerized tools/workflows, workflow scripts stored).

10. Governance & Compliance
	•	Human-in-the-loop approval required for destructive/major corrections.
	•	Data privacy/security measures documented (esp. for human data).
	•	Version-controlled repository maintained (data, code, configs, logs).
