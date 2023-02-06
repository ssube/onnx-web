.PHONY: ci pip test package-dist package-upload

onnx_env: ## create virtual env
	python -v venv onnx_env

ci: pip pip-dev lint-check
	$(MAKE) test

check-venv:
	if [ -z $${VIRTUAL_ENV+x} ]; then echo "Are you sure you want to install dependencies outside of a virtual environment?"; sleep 30; fi

pip: check-venv
	pip install -r requirements.txt

pip-dev: check-venv
	pip install -r requirements.txt

test:
	python -m coverage run -m unittest discover -s tests/
	python -m coverage html
	python -m coverage xml

package: package-dist package-upload

package-dist:
	python3 ./setup.py sdist

package-upload:
	twine upload dist/*

lint-check:
	black --check --preview onnx_web/
	isort --check-only --skip __init__.py --filter-files onnx_web
	flake8 onnx_web

lint-fix:
	black onnx_web/
	isort --skip __init__.py --filter-files onnx_web
	flake8 onnx_web

typecheck:
	mypy -m onnx_web.serve
