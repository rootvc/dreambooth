import inspect
import logging
from functools import wraps
from io import StringIO
from typing import Any

import numpy
from rich.console import Console
from rich.pretty import Pretty
from wrapt import ObjectProxy


def _prettify(obj: Any):
    if isinstance(obj, (int, float, bool, numpy.number)):
        return obj
    io = StringIO()
    console = Console(file=io, force_terminal=True)
    console.print(Pretty(obj, max_length=6, max_string=50))
    return io.getvalue().strip()


def _wrap_logger_method(fn):
    @wraps(fn)
    def wrapper(self: ObjectProxy, message: str, *args, **kwargs):
        return getattr(self.__wrapped__, fn.__name__)(
            message,
            *map(_prettify, args),
            **{k: _prettify(v) for k, v in kwargs.items()}
        )

    return wrapper


class PrettyLogger(ObjectProxy):
    def __reduce_ex__(self, _protocol: int):
        return PrettyLogger, (self.__wrapped__,)

    @_wrap_logger_method
    def info(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def trace(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def debug(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def warning(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def exception(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def error(self, message: str, *args, **kwargs):
        pass

    @_wrap_logger_method
    def critical(self, message: str, *args, **kwargs):
        pass

    def bind(self, **kwargs):
        return PrettyLogger(self.__wrapped__.bind(**kwargs))


from loguru import logger as __logger

logger = PrettyLogger(__logger)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        if (
            any(
                mod in record.name
                for mod in {"hpack", "boto", "botocore", "s3transfer"}
            )
            and level == "DEBUG"
        ):
            return
        if str(record.msg).startswith("rel_map"):
            return
        if isinstance(record.args, tuple):
            record.args = tuple(map(_prettify, record.args))
        elif isinstance(record.args, dict):
            record.args = {k: _prettify(v) for k, v in record.args.items()}

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        try:
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
        except Exception:
            self.handleError(record)
