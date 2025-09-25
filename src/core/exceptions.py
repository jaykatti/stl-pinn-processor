class STLPINNException(Exception):
    """Base exception for STL-PINN processor"""
    pass

class MeshProcessingError(STLPINNException):
    """Raised when mesh processing fails"""
    pass

class PINNModelError(STLPINNException):
    """Raised when PINN model operations fail"""
    pass

class LLMServiceError(STLPINNException):
    """Raised when LLM service operations fail"""
    pass

class ValidationError(STLPINNException):
    """Raised when validation fails"""
    pass

class FileProcessingError(STLPINNException):
    """Raised when file operations fail"""
    pass