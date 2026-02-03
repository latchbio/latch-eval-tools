import os

os.environ.setdefault("auto_reload", "false")
os.environ.setdefault("logging_mode", "console")
os.environ.setdefault("domain", "latch.bio")
os.environ.setdefault("DD_VERSION", "eval")
os.environ.setdefault("DD_SERVICE", "latch-plots-eval")
os.environ.setdefault("DD_ENV", "eval")
os.environ.setdefault("DD_AGENT_HOST", "localhost")
os.environ.setdefault("DD_TRACE_ENABLED", "false")
os.environ.setdefault("DD_PROFILING_ENABLED", "false")
os.environ.setdefault("DD_RUNTIME_METRICS_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
