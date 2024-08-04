# Makefile for Sphinx documentation

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = LLY-DML
SOURCEDIR     = source
BUILDDIR      = _build

.PHONY: help Makefile

help:
	@echo "  make html    to build the HTML documentation"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf $(BUILDDIR)/*
