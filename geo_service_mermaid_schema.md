# GEO Service Architecture - Mermaid Schema

## Class Structure and Data Flow

```mermaid
classDiagram
    %% Data Structures and Enums
    class GEODataSource {
        <<enumeration>>
        GEOPARSE
        SOFT_FILE
        SUPPLEMENTARY
        TAR_ARCHIVE
        SAMPLE_MATRICES
    }
    
    class GEODataType {
        <<enumeration>>
        SINGLE_CELL
        BULK
        MIXED
    }
    
    class DownloadStrategy {
        +bool prefer_geoparse
        +bool allow_fallback
        +int max_retries
        +int timeout_seconds
        +bool prefer_supplementary
        +bool force_tar_extraction
    }
    
    class GEOResult {
        +DataFrame data
        +Dict metadata
        +GEODataSource source
        +Dict processing_info
        +bool success
        +str error_message
    }
    
    %% Main Service Class
    class GEOService {
        +DataManagerV2 data_manager
        +Path cache_dir
        +GEODownloadManager geo_downloader
        +GEOParser geo_parser
        +DownloadStrategy download_strategy
        +Dict processing_pipelines
        
        +__init__(data_manager, cache_dir, console)
        +download_dataset(geo_id, modality_type) str
        +download_with_strategy(geo_id, strategy, data_type) GEOResult
        +fetch_metadata_only(geo_id) str
        -_try_geoparse_download(geo_id, metadata) GEOResult
        -_validate_matrices(sample_matrices) Dict
        -_concatenate_stored_samples(geo_id, stored_samples) DataFrame
    }
    
    %% External Dependencies
    class DataManagerV2 {
        +load_modality()
        +save_modality()
        +metadata_store
        +log_tool_usage()
    }
    
    class GEODownloadManager {
        +download_file()
        +download_geo_data()
    }
    
    class GEOParser {
        +parse_soft_file()
        +parse_supplementary_file()
    }
    
    %% Relationships
    GEOService --> DataManagerV2 : uses
    GEOService --> GEODownloadManager : uses
    GEOService --> GEOParser : uses
    GEOService --> DownloadStrategy : configures
    GEOService --> GEOResult : returns
    GEOResult --> GEODataSource : references
    GEOService --> GEODataType : processes
```

## Main Process Flow

```mermaid
flowchart TD
    A[DataExpert calls download_dataset] --> B{Check if metadata exists}
    B -->|No| C[Call fetch_metadata_only]
    B -->|Yes| D[Check if modality exists]
    
    C --> E{Metadata fetch successful?}
    E -->|No| F[Return error message]
    E -->|Yes| G[Store metadata in data_manager]
    
    D -->|Exists| H[Return already loaded message]
    D -->|New| I[Call download_with_strategy]
    G --> I
    
    I --> J[Determine data type from metadata]
    J --> K[Select processing pipeline]
    K --> L[Execute pipeline steps]
    
    L --> M[_try_geoparse_download]
    M --> N{GEOparse successful?}
    
    N -->|Yes| O[Get sample info]
    O --> P[Download sample matrices]
    P --> Q[Validate matrices]
    Q --> R{Is single-cell with multiple samples?}
    
    R -->|Yes| S[Store samples as AnnData]
    S --> T[Concatenate stored samples]
    T --> U[Create final modality]
    
    R -->|No| V[Concatenate matrices directly]
    V --> U
    
    N -->|No| W[Try supplementary files]
    W --> X{Supplementary files found?}
    X -->|Yes| U
    X -->|No| Y[Return failure]
    
    U --> Z[Save modality to workspace]
    Z --> AA[Return success message]
```

## Metadata Processing Flow

```mermaid
flowchart TD
    A[fetch_metadata_only called] --> B{Check metadata cache}
    B -->|Cached| C[Return formatted summary]
    B -->|Not cached| D[Try GEOparse metadata fetch]
    
    D --> E{GEOparse successful?}
    E -->|Yes| F[Extract metadata from GSE object]
    E -->|No| G[Fallback to helper services]
    
    G --> H[Use GEODownloadManager for SOFT file]
    H --> I[Use GEOParser to parse SOFT file]
    I --> J{Helper parsing successful?}
    J -->|No| K[Return error message]
    J -->|Yes| F
    
    F --> L[Validate metadata against schema]
    L --> M[Determine data type from metadata]
    M --> N[Store in data_manager.metadata_store]
    N --> O[Format metadata summary]
    O --> P[Return formatted summary to user]
```

## Sample Processing Pipeline

```mermaid
flowchart TD
    A[_try_geoparse_download] --> B[Get GSE object from GEOparse]
    B --> C[Determine if single-cell or bulk]
    C --> D[Get sample information]
    D --> E[Download sample matrices in parallel]
    
    E --> F[ThreadPoolExecutor processes samples]
    F --> G[For each sample: _download_single_sample]
    
    G --> H[Get GSM object]
    H --> I[Extract supplementary files from metadata]
    I --> J[Classify file types using regex patterns]
    J --> K{10X trio complete?}
    
    K -->|Yes| L[Download matrix, barcodes, features]
    L --> M[Parse 10X format data]
    M --> N[Create DataFrame with proper indexing]
    
    K -->|No| O{H5 file available?}
    O -->|Yes| P[Download and parse H5 file]
    P --> N
    
    O -->|No| Q{Expression file available?}
    Q -->|Yes| R[Download and parse expression file]
    R --> N
    Q -->|No| S[Return None - no usable data]
    
    N --> T[Validate matrix format]
    T --> U[Return sample matrix]
```

## File Classification System

```mermaid
flowchart TD
    A[Supplementary files found] --> B[Initialize file type patterns]
    B --> C[Extract all file URLs from metadata]
    C --> D[For each file: classify using regex patterns]
    
    D --> E[Score against pattern types]
    E --> F{Matrix file patterns?}
    F -->|Match| G[Score as 'matrix' type]
    
    E --> H{Barcodes file patterns?}
    H -->|Match| I[Score as 'barcodes' type]
    
    E --> J{Features/genes patterns?}
    J -->|Match| K[Score as 'features' type]
    
    E --> L{H5 format patterns?}
    L -->|Match| M[Score as 'h5_data' type]
    
    G --> N[Apply keyword boosts]
    I --> N
    K --> N
    M --> N
    
    N --> O[Select highest scoring file for each type]
    O --> P{10X trio complete?}
    P -->|Yes| Q[Use 10X processing pipeline]
    P -->|No| R[Use alternative processing pipeline]
```

## Data Storage and Integration

```mermaid
flowchart TD
    A[Processed matrices ready] --> B{Single-cell with multiple samples?}
    
    B -->|Yes| C[Store each sample as individual AnnData]
    C --> D[Create modality for each sample]
    D --> E[Save individual samples to workspace]
    E --> F[Concatenate using anndata.concat]
    F --> G{Use intersecting genes only?}
    G -->|Yes| H[Inner join - common genes only]
    G -->|No| I[Outer join - all genes, fill missing with 0]
    
    B -->|No| J[Concatenate matrices directly]
    J --> K[Add batch information]
    
    H --> L[Create final combined dataset]
    I --> L
    K --> L
    
    L --> M[Determine appropriate adapter]
    M --> N{High gene count?}
    N -->|Yes| O[Use transcriptomics_single_cell adapter]
    N -->|No| P[Use transcriptomics_bulk adapter]
    
    O --> Q[Load as modality in DataManagerV2]
    P --> Q
    Q --> R[Save to workspace]
    R --> S[Log tool usage]
    S --> T[Auto-save state]
    T --> U[Return success message]
```

## Error Handling and Fallback Strategy

```mermaid
flowchart TD
    A[Primary GEOparse attempt] --> B{Success?}
    B -->|No| C[Log warning and try fallback]
    B -->|Yes| D[Process normally]
    
    C --> E[Use GEODownloadManager]
    E --> F{Download successful?}
    F -->|No| G[Try next fallback method]
    F -->|Yes| H[Use GEOParser]
    
    H --> I{Parse successful?}
    I -->|No| G
    I -->|Yes| J[Continue with parsed data]
    
    G --> K{Max retries reached?}
    K -->|No| L[Retry with different strategy]
    K -->|Yes| M[Return GEOResult with failure]
    
    L --> E
    J --> D
    D --> N[Return GEOResult with success]
```

## Key Design Patterns

1. **Strategy Pattern**: `DownloadStrategy` configures download behavior
2. **Template Method**: `download_with_strategy` defines the algorithm skeleton
3. **Factory Pattern**: Processing pipelines created based on data type
4. **Observer Pattern**: Progress tracking through console and logging
5. **Facade Pattern**: `GEOService` provides simplified interface to complex subsystems
6. **Command Pattern**: Each pipeline step is a callable function
7. **Builder Pattern**: `GEOResult` accumulates processing information

## Data Flow Summary

The service follows a layered approach:
1. **Metadata Layer**: Fetch and validate metadata first
2. **Strategy Layer**: Determine download approach based on data type
3. **Processing Layer**: Execute pipeline steps with fallbacks
4. **Validation Layer**: Ensure data quality and format
5. **Storage Layer**: Integrate with DataManagerV2 for persistence
6. **Integration Layer**: Provide seamless access to processed data
