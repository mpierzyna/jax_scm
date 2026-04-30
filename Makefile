
SRC_FILES  := $(shell find src/scm -name '*.py')
TEST_FILES := $(shell find tests -name '*.py')

.PHONY: tests coverage badges

tests coverage: docs/coverage.xml

docs/junit.xml: docs/coverage.xml

docs/coverage.xml: $(SRC_FILES) $(TEST_FILES)
	uv run pytest tests/ --junitxml=docs/junit.xml --cov=src/scm --cov-report=html --cov-report=xml:docs/coverage.xml

docs/coverage-badge.svg: docs/coverage.xml
	uv run genbadge coverage -i $< -o $@

docs/test-badge.svg: docs/junit.xml
	uv run genbadge tests -i $< -o $@

badges: docs/coverage-badge.svg docs/test-badge.svg