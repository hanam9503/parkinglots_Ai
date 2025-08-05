"""
Custom Exceptions cho Enhanced Parking System
Định nghĩa các exception tùy chỉnh cho hệ thống
"""

class ParkingSystemException(Exception):
    """Base exception for parking system"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"[{self.error_code}] {self.message} - Details: {self.details}"
        return f"[{self.error_code}] {self.message}"

# Model-related exceptions
class ModelException(ParkingSystemException):
    """Base exception for model-related errors"""
    pass

class ModelLoadException(ModelException):
    """Exception raised when model loading fails"""
    
    def __init__(self, model_name: str, model_path: str, reason: str = None):
        message = f"Failed to load model '{model_name}' from '{model_path}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "model_path": model_path, "reason": reason}
        )

class ModelInferenceException(ModelException):
    """Exception raised during model inference"""
    
    def __init__(self, model_name: str, reason: str = None):
        message = f"Model inference failed for '{model_name}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="MODEL_INFERENCE_ERROR",
            details={"model_name": model_name, "reason": reason}
        )

# Detection-related exceptions
class DetectionException(ParkingSystemException):
    """Base exception for detection-related errors"""
    pass

class VehicleDetectionException(DetectionException):
    """Exception raised during vehicle detection"""
    
    def __init__(self, reason: str = None, frame_info: dict = None):
        message = "Vehicle detection failed"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="VEHICLE_DETECTION_ERROR",
            details={"reason": reason, "frame_info": frame_info or {}}
        )

class PlateDetectionException(DetectionException):
    """Exception raised during license plate detection"""
    
    def __init__(self, reason: str = None, vehicle_info: dict = None):
        message = "License plate detection failed"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="PLATE_DETECTION_ERROR",
            details={"reason": reason, "vehicle_info": vehicle_info or {}}
        )

# Processing-related exceptions
class ProcessingException(ParkingSystemException):
    """Base exception for processing-related errors"""
    pass

class ImageProcessingException(ProcessingException):
    """Exception raised during image processing"""
    
    def __init__(self, operation: str, reason: str = None, image_info: dict = None):
        message = f"Image processing failed during '{operation}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            details={"operation": operation, "reason": reason, "image_info": image_info or {}}
        )

class OCRException(ProcessingException):
    """Exception raised during OCR processing"""
    
    def __init__(self, reason: str = None, image_info: dict = None):
        message = "OCR processing failed"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="OCR_ERROR",
            details={"reason": reason, "image_info": image_info or {}}
        )

class EnhancementException(ProcessingException):
    """Exception raised during image enhancement"""
    
    def __init__(self, enhancement_type: str, reason: str = None):
        message = f"Image enhancement failed for '{enhancement_type}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="ENHANCEMENT_ERROR",
            details={"enhancement_type": enhancement_type, "reason": reason}
        )

# Configuration-related exceptions
class ConfigurationException(ParkingSystemException):
    """Base exception for configuration-related errors"""
    pass

class InvalidConfigurationException(ConfigurationException):
    """Exception raised for invalid configuration"""
    
    def __init__(self, config_key: str, value: str = None, expected: str = None):
        message = f"Invalid configuration for '{config_key}'"
        if value:
            message += f" (value: {value})"
        if expected:
            message += f" (expected: {expected})"
        
        super().__init__(
            message=message,
            error_code="INVALID_CONFIG",
            details={"config_key": config_key, "value": value, "expected": expected}
        )

class MissingConfigurationException(ConfigurationException):
    """Exception raised for missing configuration"""
    
    def __init__(self, config_key: str, config_file: str = None):
        message = f"Missing required configuration: '{config_key}'"
        if config_file:
            message += f" in file '{config_file}'"
        
        super().__init__(
            message=message,
            error_code="MISSING_CONFIG",
            details={"config_key": config_key, "config_file": config_file}
        )

# Sync and communication exceptions
class SyncException(ParkingSystemException):
    """Base exception for synchronization errors"""
    pass

class ServerConnectionException(SyncException):
    """Exception raised for server connection issues"""
    
    def __init__(self, server_url: str, reason: str = None, status_code: int = None):
        message = f"Failed to connect to server '{server_url}'"
        if reason:
            message += f": {reason}"
        if status_code:
            message += f" (HTTP {status_code})"
        
        super().__init__(
            message=message,
            error_code="SERVER_CONNECTION_ERROR",
            details={"server_url": server_url, "reason": reason, "status_code": status_code}
        )

class DataSyncException(SyncException):
    """Exception raised during data synchronization"""
    
    def __init__(self, operation: str, reason: str = None, data_info: dict = None):
        message = f"Data synchronization failed for '{operation}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="DATA_SYNC_ERROR",
            details={"operation": operation, "reason": reason, "data_info": data_info or {}}
        )

class OfflineQueueException(SyncException):
    """Exception raised for offline queue operations"""
    
    def __init__(self, operation: str, reason: str = None, queue_size: int = None):
        message = f"Offline queue operation failed: '{operation}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="OFFLINE_QUEUE_ERROR",
            details={"operation": operation, "reason": reason, "queue_size": queue_size}
        )

# Data validation exceptions
class ValidationException(ParkingSystemException):
    """Base exception for data validation errors"""
    pass

class InvalidEventDataException(ValidationException):
    """Exception raised for invalid event data"""
    
    def __init__(self, field: str = None, value: str = None, reason: str = None):
        message = "Invalid event data"
        if field:
            message += f" for field '{field}'"
        if value:
            message += f" (value: {value})"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="INVALID_EVENT_DATA",
            details={"field": field, "value": value, "reason": reason}
        )

class InvalidStatusDataException(ValidationException):
    """Exception raised for invalid status data"""
    
    def __init__(self, field: str = None, value: str = None, reason: str = None):
        message = "Invalid status data"
        if field:
            message += f" for field '{field}'"
        if value:
            message += f" (value: {value})"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="INVALID_STATUS_DATA",
            details={"field": field, "value": value, "reason": reason}
        )

class InvalidPlateException(ValidationException):
    """Exception raised for invalid license plate"""
    
    def __init__(self, plate_text: str, reason: str = None):
        message = f"Invalid license plate: '{plate_text}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="INVALID_PLATE",
            details={"plate_text": plate_text, "reason": reason}
        )

# File and I/O exceptions
class FileException(ParkingSystemException):
    """Base exception for file-related errors"""
    pass

class FileNotFoundException(FileException):
    """Exception raised when file is not found"""
    
    def __init__(self, file_path: str, file_type: str = None):
        message = f"File not found: '{file_path}'"
        if file_type:
            message += f" ({file_type})"
        
        super().__init__(
            message=message,
            error_code="FILE_NOT_FOUND",
            details={"file_path": file_path, "file_type": file_type}
        )

class FileWriteException(FileException):
    """Exception raised when file write fails"""
    
    def __init__(self, file_path: str, reason: str = None):
        message = f"Failed to write file: '{file_path}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="FILE_WRITE_ERROR",
            details={"file_path": file_path, "reason": reason}
        )

class FileReadException(FileException):
    """Exception raised when file read fails"""
    
    def __init__(self, file_path: str, reason: str = None):
        message = f"Failed to read file: '{file_path}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="FILE_READ_ERROR",
            details={"file_path": file_path, "reason": reason}
        )

# System and resource exceptions
class SystemException(ParkingSystemException):
    """Base exception for system-related errors"""
    pass

class InsufficientResourcesException(SystemException):
    """Exception raised when system resources are insufficient"""
    
    def __init__(self, resource_type: str, required: str = None, available: str = None):
        message = f"Insufficient {resource_type}"
        if required and available:
            message += f" (required: {required}, available: {available})"
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_RESOURCES",
            details={"resource_type": resource_type, "required": required, "available": available}
        )

class TimeoutException(SystemException):
    """Exception raised when operation times out"""
    
    def __init__(self, operation: str, timeout_seconds: float = None):
        message = f"Operation timed out: '{operation}'"
        if timeout_seconds:
            message += f" (timeout: {timeout_seconds}s)"
        
        super().__init__(
            message=message,
            error_code="OPERATION_TIMEOUT",
            details={"operation": operation, "timeout_seconds": timeout_seconds}
        )

class InitializationException(SystemException):
    """Exception raised during system initialization"""
    
    def __init__(self, component: str, reason: str = None):
        message = f"Failed to initialize '{component}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="INITIALIZATION_ERROR",
            details={"component": component, "reason": reason}
        )

# Utility functions for exception handling
def handle_exception(exception: Exception, logger=None, reraise: bool = True):
    """Universal exception handler"""
    
    if isinstance(exception, ParkingSystemException):
        error_msg = str(exception)
        error_code = exception.error_code
        details = exception.details
    else:
        error_msg = f"Unexpected error: {str(exception)}"
        error_code = "UNEXPECTED_ERROR"
        details = {"exception_type": type(exception).__name__}
    
    if logger:
        logger.error(f"{error_code}: {error_msg}")
        if details:
            logger.debug(f"Error details: {details}")
    else:
        print(f"ERROR [{error_code}]: {error_msg}")
        if details:
            print(f"Details: {details}")
    
    if reraise:
        if isinstance(exception, ParkingSystemException):
            raise exception
        else:
            raise ParkingSystemException(
                message=error_msg,
                error_code=error_code,
                details=details
            )

def create_error_response(exception: Exception) -> dict:
    """Create standardized error response"""
    
    if isinstance(exception, ParkingSystemException):
        return {
            "success": False,
            "error": {
                "code": exception.error_code,
                "message": exception.message,
                "details": exception.details
            }
        }
    else:
        return {
            "success": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": str(exception),
                "details": {"exception_type": type(exception).__name__}
            }
        }

# Context manager for exception handling
class ExceptionContext:
    """Context manager for handling exceptions with logging"""
    
    def __init__(self, operation_name: str, logger=None, reraise: bool = True):
        self.operation_name = operation_name
        self.logger = logger
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.logger:
                self.logger.error(f"Exception in operation '{self.operation_name}': {exc_val}")
            
            if self.reraise:
                return False  # Re-raise the exception
            else:
                return True  # Suppress the exception
        
        return False

# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test custom exceptions
    try:
        raise ModelLoadException("YOLOv8", "/path/to/model.pt", "File not found")
    except ParkingSystemException as e:
        print(f"Caught custom exception: {e}")
        print(f"Error code: {e.error_code}")
        print(f"Details: {e.details}")
    
    # Test exception handler
    try:
        raise ValueError("This is a standard Python exception")
    except Exception as e:
        try:
            handle_exception(e, logger, reraise=True)
        except ParkingSystemException as wrapped_e:
            print(f"Wrapped exception: {wrapped_e}")
    
    # Test context manager
    with ExceptionContext("test_operation", logger, reraise=False):
        raise RuntimeError("This will be caught and logged")
    
    print("Program continues after handled exception")
    
    # Test error response creation
    try:
        raise ServerConnectionException("http://localhost:5000", "Connection refused", 503)
    except Exception as e:
        error_response = create_error_response(e)
        print(f"Error response: {error_response}")