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

all: pdf html

.PHONY: apidoc
apidoc:
	sphinx-apidoc $(SPHINX_APIDOC_ARGS) $(PWD)/src/BIA_G8

html: apidoc
	sphinx-build -M html $(CURDIR) $(BUILD_DIR) $(SPHINX_OPTS)

pdf: apidoc
	sphinx-build -M latexpdf $(CURDIR) $(BUILD_DIR) $(SPHINX_OPTS)

clean:
	$(RM) $(BUILD_DIR)

distclean: clean
