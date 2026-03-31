The previous command failed {{ failure_reason }}
{{ container_action }}

Files under `{{ workspace_dir }}` were preserved because that directory is bind-mounted. Any packages, background processes, or shell state created elsewhere in the old container were lost.

Continue from the preserved workspace and prefer smaller-memory, smaller-scope commands before retrying.
