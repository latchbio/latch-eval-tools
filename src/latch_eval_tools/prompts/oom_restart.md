Your previous session was interrupted by an Out-of-Memory (OOM) event.
{{ container_action }}

Session state under `{{ state_dir }}` was preserved via the bind mount, so you can continue from the previous conversation state.

Please reduce memory usage in the next steps: prefer streaming or chunked processing, avoid loading large datasets entirely into memory, clean up intermediate artifacts when possible, and avoid starting memory-heavy subprocesses unless necessary.
