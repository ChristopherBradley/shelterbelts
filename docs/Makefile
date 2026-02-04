.PHONY: help Makefile

help:
	@echo "Targets for Sphinx documentation:"
	@echo "  html        to make standalone HTML files"
	@echo "  doctest     run doctest on all docstrings"
	@echo "  clean       remove build artifacts"

Makefile:

SPHINXBUILD   = conda run -n shelterbelts_pdal sphinx-build
BUILDDIR      = build
SOURCEDIR     = source

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"

doctest:
	cd .. && conda run -n shelterbelts_pdal pytest --doctest-modules src/shelterbelts/indices/tree_categories.py src/shelterbelts/indices/shelter_categories.py src/shelterbelts/indices/buffer_categories.py src/shelterbelts/indices/cover_categories.py -v


clean:
	rm -rf $(BUILDDIR)
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
