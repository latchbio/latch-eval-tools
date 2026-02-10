build:
  rm -rf dist
  uv build

publish:
  uv publish --token $(<credentials/pypi_token)
  rm -rf dist

install:
  uv pip install -e .