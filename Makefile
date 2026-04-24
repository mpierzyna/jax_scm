
tests:
coverage:
docs/junit.xml:
docs/coverage.xml: src/scm/ tests/
	uv run pytest tests/ --junitxml=docs/junit.xml --cov=src/scm --cov-report=html --cov-report=xml:docs/coverage.xml

docs/coverage-badge.svg: docs/coverage.xml
	uv run genbadge coverage -i $< -o $@

docs/test-badge.svg: docs/junit.xml
	uv run genbadge tests -i $< -o $@

badges: docs/coverage-badge.svg docs/test-badge.svg