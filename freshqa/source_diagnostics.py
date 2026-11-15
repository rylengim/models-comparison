"""Sanitized, versioned source observations; no inference about factual correctness."""
from dataclasses import dataclass

import requests

VERSION = 'source-diagnostics-v1'


def diagnostic(stage, outcome, *, error_kind=None, http_status=None, retryable=None):
    return dict(version=VERSION, stage=stage, outcome=outcome, error_kind=error_kind,
                http_status=http_status, retryable=retryable)


@dataclass(frozen=True)
class SourceBody:
    text: str
    http_status: int


class SourceFailure(Exception):
    """Messages are chosen by the boundary, never copied from request exceptions."""
    def __init__(self, message, *, stage, kind, http_status=None, retryable=False):
        super().__init__(message)
        self.diagnostic = diagnostic(stage, 'error', error_kind=kind,
                                     http_status=http_status, retryable=retryable)

    def result(self):
        return {'error': str(self), 'diagnostic': self.diagnostic}


def request_failure(error, *, stage, http_status=None):
    if isinstance(error, requests.exceptions.SSLError):
        kind, retryable = 'tls_error', False
    elif isinstance(error, requests.Timeout):
        kind, retryable = 'timeout', True
    elif isinstance(error, requests.HTTPError):
        kind = 'http_error'
        retryable = None if http_status is None else http_status in {408, 425, 429, 500, 502, 503, 504}
    elif isinstance(error, requests.ConnectionError):
        kind, retryable = 'connection_error', True
    else:
        kind, retryable = 'request_error', None
    return SourceFailure('Source request failed: ' + kind, stage=stage, kind=kind,
                         http_status=http_status, retryable=retryable)
