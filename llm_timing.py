import json
import logging
from contextlib import contextmanager
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger("llm_timing")

@contextmanager
def track_llm_call(operation: str, model: str, evidence=None):
    call_id = uuid4().hex
    started = perf_counter()
    status = "success"

    fields = {}
    try:
        yield fields
    except BaseException as exc:
        fields["error_type"] = type(exc).__name__
        status = "error"
        raise
    finally:
        record = {
            "event": "llm_call",
            "call_id": call_id,
            "operation": operation,
            "model": model,
            "status": status,
            "duration_ms": round(
                (perf_counter() - started) * 1000, 2
            ),
        }
        logger.info(json.dumps(record))
        if evidence is not None:
            evidence.call({**record, **fields})