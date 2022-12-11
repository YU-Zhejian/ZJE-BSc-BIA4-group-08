.PHONY: dist
dist:
	PYTHONPATH=$(CURDIR) python -m build

.PHONY: doc
doc:
	$(MAKE) -C doc

.PHONY: cleandoc
cleandoc:
	$(MAKE) -C doc clean
	$(MAKE) doc

.PHONY: serve-doc
serve-doc:
	python -m http.server -d doc/_build/html

.PHONY: test
test:
	PYTHONPATH=src pytest .
	rm -f .coverage.*

.PHONY: pytype
pytype:
	 pytype --config=pytype.cfg src/BIA_G8
