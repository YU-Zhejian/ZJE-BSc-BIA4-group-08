SPHINX_OPTS := -j auto

SPHINX_APIDOC_ARGS := --separate \
	--no-toc \
	--module-first \
	--maxdepth 2 \
	--force \
	--output-dir $(CURDIR)/_apidoc \
	--doc-project 'API Docs'

BUILD_DIR := $(CURDIR)/_build

.PHONY: all
all: html

.PHONY: apidoc
apidoc:
	sphinx-apidoc $(SPHINX_APIDOC_ARGS) $(CURDIR)/../src/BIA_G8

html: apidoc
	sphinx-build -M html $(CURDIR) $(BUILD_DIR) $(SPHINX_OPTS)

clean:
	rm -rf $(CURDIR)/_build $(CURDIR)/_apidoc $(CURDIR)/_root

distclean: clean
