"""Project-specific exception hierarchy for translation workflows."""


class BookTranslatorError(Exception):
    """Base exception for book translation failures."""


class ConfigurationError(BookTranslatorError):
    """Raised when required runtime configuration is missing."""


class FileProcessingError(BookTranslatorError):
    """Raised when an input or output file cannot be processed."""


class OCRProcessingError(BookTranslatorError):
    """Raised when PDF image conversion or OCR fails."""


class TranslationError(BookTranslatorError):
    """Raised when translation with the configured LLM fails."""
