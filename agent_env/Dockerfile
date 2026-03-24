FROM python:3.12-slim-bookworm

SHELL [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-c" \
]

ENV TZ='Etc/UTC'
ENV LANG='en_US.UTF-8'

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies (includes libs needed by R packages and Python native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg git \
    ripgrep vim sudo nodejs npm \
    build-essential gfortran pkg-config \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    libhdf5-dev libgeos-dev libproj-dev libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV IS_SANDBOX=1

# OpenAI Codex CLI
RUN npm config set prefix /root/.local && \
    npm install -g @openai/codex@0.116.0

# Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"
RUN claude install 2.1.76

# R (from CRAN for a recent release on Bookworm)
RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7" \
        | gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" \
        > /etc/apt/sources.list.d/cran.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

    
# Python environment — agent-env deps
# Build context: latch-eval-tools/env/  (docker build -f Dockerfile .)
COPY . /root/

WORKDIR /root

RUN uv sync --no-dev

ENV VIRTUAL_ENV=/root/.venv
ENV PATH="/root/.venv/bin:/root/.local/bin:$PATH"
RUN printf '%s\n' \
    'export VIRTUAL_ENV=/root/.venv' \
    'export PATH="/root/.venv/bin:/root/.local/bin:$PATH"' \
    > /root/.bashrc && \
    printf '%s\n' \
    'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' \
    > /root/.bash_profile
