# Implementation Plan

## Overview
Refactor the GEO service to create a modular, robust system that integrates GEOparse with fallback mechanisms using helper files for comprehensive GEO data downloading and processing.

## Scope
The implementation creates a layered architecture where GEOparse is the primary method, with sophisticated fallback mechanisms using geo_downloader.py and geo_parser.py helper files. The system handles both single-cell and bulk RNA-seq data with comprehensive error recovery and systematic naming conventions.

## Types
Data structure and interface definitions for the modular GEO service system.

**Core Data Types:**
- `GEODataSource`: Enum defining data source types (GEOparse, SOFT, Supplementary, TAR)
- `GEODataType`: Enum for data types (SingleCell, Bulk, Mixed)
- `DownloadStrategy`: Configuration class for download preferences and fallback options
- `GEOResult`: Result wrapper containing data, metadata, and processing information
- `ProcessingPipeline`: Pipeline configuration for different data scenarios

**Interface Definitions:**
```python
class GEODataSource(Enum):
    GEOPARSE = "geoparse"
    SOFT_FILE = "soft_file" 
    SUPPLEMENTARY = "supplementary"
    TAR_ARCHIVE = "tar_archive"
    SAMPLE_MATRICES = "sample_matrices"

class GEODataType(Enum):
    SINGLE_CELL = "single_cell"
    BULK = "bulk"
    MIXED = "mixed"

class DownloadStrategy:
    prefer_geoparse: bool = True
    allow_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
```

## Files
Detailed file modifications and new file creation requirements.

**Modified Files:**
- `lobster/tools/geo_service.py`: Complete refactoring with modular architecture
  - Integrate geo_downloader.GEODownloadManager for fallback downloading
  - Integrate geo_parser.GEOParser for fallback parsing
  - Implement layered download strategy with proper error handling
  - Add specialized methods for each of the 6 scenarios
  - Maintain backward compatibility with existing DataManagerV2 integration

**Helper Files Integration:**
- `lobster/tools/geo_downloader.py`: Enhanced with better error reporting
  - Add integration points for geo_service callback
  - Improve download progress reporting
  - Add validation for downloaded files

- `lobster/tools/geo_parser.py`: Enhanced parsing capabilities  
  - Add better format detection
  - Improve error handling and validation
  - Add support for additional single-cell formats

**Configuration Updates:**
- Update import statements to include helper classes
- Add logging configuration for comprehensive tracking
- Add settings for download timeouts and retry logic

## Functions
Function-level modifications and new function implementations.

**New Core Functions in GEOService:**
- `download_with_strategy()`: Master function implementing layered download approach
- `download_single_cell_sample()`: Specialized single-cell sample downloading (Scenario 4)
- `download_bulk_dataset()`: Enhanced bulk data downloading (Scenario 5)  
- `process_supplementary_tar_files()`: TAR file processing fallback (Scenario 6)
- `_detect_and_route_data_type()`: Smart routing based on metadata analysis
- `_create_fallback_pipeline()`: Dynamic fallback pipeline creation
- `_integrate_helper_services()`: Helper service integration and coordination

**Enhanced Existing Functions:**
- `fetch_metadata_only()`: Enhanced with helper file fallback (Scenario 1)
- `download_dataset()`: Complete refactoring with scenario routing (Scenario 2)
- `_process_supplementary_files()`: Integration with geo_parser
- `_validate_geo_metadata()`: Enhanced validation with data type detection

**Helper Integration Functions:**
- `_download_with_geo_downloader()`: Wrapper for GEODownloadManager integration
- `_parse_with_geo_parser()`: Wrapper for GEOParser integration
- `_coordinate_fallback_strategy()`: Orchestrate fallback mechanisms

## Classes
Class structure modifications and new class definitions.

**Enhanced GEOService Class:**
- Add `geo_downloader: GEODownloadManager` attribute for fallback downloading
- Add `geo_parser: GEOParser` attribute for fallback parsing
- Add `download_strategy: DownloadStrategy` for configuration
- Add `processing_pipeline: Dict[str, List[Callable]]` for scenario routing
- Maintain existing `data_manager: DataManagerV2` integration
- Add comprehensive error tracking and recovery mechanisms

**New Helper Integration Classes:**
- `GEOServiceCoordinator`: Coordinates between main service and helper files
- `ScenarioRouter`: Routes download requests to appropriate processing pipeline
- `FallbackManager`: Manages fallback strategies and error recovery

**Enhanced Error Handling:**
- Custom exception classes for different failure modes
- Comprehensive logging at each processing stage
- Graceful degradation when components fail

## Dependencies
Package and library requirements for the enhanced system.

**Required Imports:**
```python
# Existing imports (maintained)
import json, os, re, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union, Callable
import numpy as np
import pandas as pd
try:
    import GEOparse
except ImportError:
    GEOparse = None

# New imports for helper integration
from lobster.tools.geo_downloader import GEODownloadManager
from lobster.tools.geo_parser import GEOParser
from enum import Enum
from dataclasses import dataclass
import time
from contextlib import contextmanager
```

**External Dependencies (no changes):**
- GEOparse: Primary GEO data access library
- pandas, numpy: Data manipulation
- requests: HTTP downloading
- scipy: Sparse matrix handling for single-cell data

## Testing
Testing approach and validation strategies for the modular system.

**Integration Testing:**
- Test each of the 6 scenarios independently
- Test fallback mechanisms under simulated failures
- Test data consistency across different download methods
- Verify DataManagerV2 integration maintains functionality

**Validation Testing:**
- Single-cell data format validation (10X, h5ad, etc.)
- Bulk data format validation
- Metadata consistency checks
- Processing pipeline integrity

**Error Recovery Testing:**
- Network failure simulation
- Partial download recovery
- Corrupted file handling
- GEOparse library unavailability

## Implementation Order
Sequential steps for implementing the modular GEO service.

**Phase 1: Core Integration (Steps 1-3)**
1. **Import Integration**: Add helper file imports and initialize integration points
2. **Basic Fallback**: Implement basic fallback from GEOparse to helper files
3. **Error Handling**: Add comprehensive error handling and logging framework

**Phase 2: Scenario Implementation (Steps 4-6)**  
4. **Scenario Routing**: Implement smart routing for different data types and scenarios
5. **Single-Cell Enhancement**: Implement specialized single-cell download methods (Scenarios 4, 6)
6. **Bulk Data Enhancement**: Refine bulk data downloading with helper integration (Scenario 5)

**Phase 3: Advanced Features (Steps 7-9)**
7. **TAR Processing**: Implement robust TAR supplementary file processing
8. **Pipeline Coordination**: Add pipeline coordination and strategy management
9. **DataManagerV2 Sync**: Ensure full compatibility with DataManagerV2 features

**Phase 4: Validation & Testing (Steps 10-12)**
10. **Integration Testing**: Test all scenarios with real GEO datasets
11. **Performance Optimization**: Optimize download speeds and memory usage
12. **Documentation**: Update docstrings and add usage examples
