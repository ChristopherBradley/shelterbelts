.PHONY: help doctest html clean test

help:
	@echo "Common project tasks:"
	@echo "  test        run pytest test suite"
	@echo "  doctest     run doctest on all docstrings"
	@echo "  html        build Sphinx HTML documentation"
	@echo "  clean       remove build artifacts"

test:
	conda run -n shelterbelts_pdal pytest tests/ -v

doctest:
	$(MAKE) -C docs doctest

html:
	$(MAKE) -C docs html

clean:
	$(MAKE) -C docs clean
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
