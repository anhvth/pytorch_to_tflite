.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard .//*.ipynb)

all: pytorch_to_tflite docs git

pytorch_to_tflite: $(SRC)
	nbdev_build_lib
	touch pytorch_to_tflite

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi 
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

git: 
	git add -A && git commit -v && git push