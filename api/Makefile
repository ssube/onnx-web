.PHONY: ci check-venv pip pip-dev lint-check lint-fix test typecheck package package-dist package-upload

onnx_env: ## create virtual env
	python -v venv onnx_env

ci: pip pip-dev lint-check
	$(MAKE) test

check-venv:
	if [ -z $${VIRTUAL_ENV+x} ]; then echo "Are you sure you want to install dependencies outside of a virtual environment?"; sleep 30; fi

pip: check-venv
	pip install -r requirements.txt

pip-dev: check-venv
	pip install -r requirements/dev.txt

test:
	python -m coverage erase
	python -m coverage run -m unittest discover -s tests/
	python -m coverage html -i
	python -m coverage xml -i

package: package-dist package-upload

package-dist:
	python3 ./setup.py sdist

package-upload:
	twine upload dist/*

lint-check:
	black --check onnx_web/
	isort --check-only --skip __init__.py --filter-files onnx_web
	flake8 onnx_web

lint-fix:
	black onnx_web/
	isort --skip __init__.py --filter-files onnx_web
	flake8 onnx_web

typecheck:
	mypy onnx_web
