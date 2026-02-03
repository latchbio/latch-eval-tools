build:
  rm -rf dist
  uv build

publish:
  uv publish --token $(<credentials/pypi_token)
  rm -rf dist
