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


.PHONY: test
test:
	PYTHONPATH=src pytest .
	rm -f .coverage.*

.PHONY: mypy
mypy:
	mypy --config-file pyproject.toml -p BIA_G8
