.PHONY: help Makefile

help:
	@echo "Targets for Sphinx documentation:"
	@echo "  html        to make standalone HTML files"
	@echo "  clean       remove build artifacts"

Makefile:

SPHINXBUILD   = sphinx-build
BUILDDIR      = build
SOURCEDIR     = source

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR)
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
