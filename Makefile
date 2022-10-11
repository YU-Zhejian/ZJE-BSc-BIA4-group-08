.PHONY: dist
dist:
	python setup.py sdist bdist_wheel

.PHONY: doc
doc:
	$(MAKE) -C doc
