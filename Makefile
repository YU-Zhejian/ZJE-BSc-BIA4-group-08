.PHONY: dist
dist:
	python setup.py sdist bdist_wheel

.PHONY: doc
doc:
	$(MAKE) -C doc

.PHONY: test
test:
	PYTHONPATH=src pytest .
	rm -f .coverage.*
