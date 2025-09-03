from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLobsterClient(ABC):
    """Abstract base class for Lobster clients (local and cloud versions)"""
    
    @abstractmethod
    def query(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Process a user query and return the result"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the client"""
        pass

class BaseDataManager(ABC):
    """Abstract base class for data managers"""
    
    @abstractmethod
    def has_data(self) -> bool:
        """Check if any data is loaded"""
        pass
    
    @abstractmethod
    def load_modality(self, name: str, source: Any, adapter: str, **kwargs) -> Any:
        """Load a data modality using the specified adapter"""
        pass
    
    @abstractmethod
    def get_modality(self, name: str) -> Any:
        """Get a loaded modality by name"""
        pass
    
    @abstractmethod
    def list_modalities(self) -> list:
        """List all loaded modalities"""
        pass
