.PHONY: dist
dist:
	python -m build

.PHONY: doc
doc:
	bash notebook-checkout.sh --execute
	$(MAKE) -C doc

.PHONY: cleandoc
cleandoc:
	$(MAKE) -C doc clean
	$(MAKE) doc

.PHONY:
serve-doc:
	python -m http.server -d doc/_build/html

.PHONY: test
test:
	PYTHONPATH=src pytest .
	rm -f .coverage.*
