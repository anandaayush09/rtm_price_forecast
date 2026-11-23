# Command line and app for the day-ahead forecast. No data goes into the image: the window table,
# the saved model, the calibration table and the forecasts are read from and written to the
# mounted data folder (see docker-compose.yml).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image does not have.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /work

# Dependencies first, so a change in the code does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
# Only what the daily forecast and the app need: the benchmark libraries stay out.
RUN uv sync --locked --no-default-groups --group app --no-install-project

COPY src ./src
COPY app ./app
COPY config ./config
COPY .streamlit ./.streamlit
COPY params.yaml ./
RUN uv sync --locked --no-default-groups --group app

RUN useradd --create-home --uid 1000 forecast && mkdir -p data && chown forecast /work/data
USER forecast

ENTRYPOINT ["rtm-forecast"]
CMD ["--help"]
