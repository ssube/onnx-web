.PHONY: ci pip test package-dist package-upload

onnx_env: ## create virtual env
	python -v venv onnx_env

ci: pip
	pip install coverage
	$(MAKE) test

pip:
	if [ -z $${VIRTUAL_ENV+x} ]; then echo "Are you sure you want to install dependencies outside of a virtual environment?"; sleep 30; fi
	pip install -r requirements.txt

test:
	python -m coverage run -m unittest discover -s tests/
	python -m coverage html
	python -m coverage xml

package-dist:
	python ./setup.py sdist

package-upload:
	twine upload dist/*
