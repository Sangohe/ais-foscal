import sys
from typing import Any, Union
from datetime import datetime
from absl.logging import PythonFormatter, converter

absl_severity_mapping = {
    "I": "INFO",
    "W": "WARNING",
    "E": "ERROR",
    "F": "CRITICAL",
}


class CustomPythonFormatter(PythonFormatter):
    def format(self, record):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d at %H:%M:%S")

        severity = converter.get_initial_for_level(record.levelno)
        severity = absl_severity_mapping[severity]
        prefix = f"[{severity} {date_time} {record.filename}:{record.lineno}] "
        msg = prefix + super(PythonFormatter, self).format(record)
        return msg


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and
    optionally force flushing on both stdout and the file.

    Taken from: https://github.com/NVlabs/stylegan3"""

    def __init__(
        self, file_name: str = None, file_mode: str = "w", should_flush: bool = True
    ):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
        if len(text) == 0:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
