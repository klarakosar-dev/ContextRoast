class ContextRoastError(Exception):
    """An actionable error safe to display without a traceback."""


class DatasetError(ContextRoastError):
    """Invalid or unreadable evaluation data."""


class ProviderError(ContextRoastError):
    """A configuration, transport, or response-envelope failure."""
