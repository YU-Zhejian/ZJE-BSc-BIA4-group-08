.PHONY: dist
dist:
	python -m build

.PHONY: doc
doc:
	bash notebook-checkout.sh --execute
	$(MAKE) -C doc clean
	$(MAKE) -C doc

.PHONY: test
test:
	PYTHONPATH=src pytest .
	rm -f .coverage.*
