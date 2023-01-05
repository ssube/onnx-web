.PHONY: build ci clean docs docs-local lint package run test

# JS targets
node_modules: deps

ci: deps lint build-shebang test

clean:
	rm -rf node_modules/
	rm -rf out/

deps:
	yarn install

docs:
	yarn api-extractor run -c .api-extractor.json
	yarn api-documenter markdown -i out/api -o docs/api

docs-local:
	yarn api-extractor run -c .api-extractor.json --local
	yarn api-documenter markdown -i out/api -o docs/api

build: deps
	yarn tsc

build-shebang: build
	sed -i '1s;^;#! /usr/bin/env node\n\n;g' $(shell pwd)/out/src/index.js
	chmod ug+x out/src/index.js

bundle: build
	node esbuild.js
	cp -v src/index.html out/

COVER_OPTS := --all \
		--exclude ".eslintrc.js" \
		--exclude "docs/**" \
		--exclude "out/coverage/**" \
		--exclude "vendor/**" \
		--reporter=text-summary \
		--reporter=lcov \
		--reporter=cobertura \
		--report-dir=out/coverage

MOCHA_OPTS := --async-only \
		--check-leaks \
		--forbid-only \
		--recursive \
		--require source-map-support/register \
		--require out/test/setup.js \
		--sort

lint: deps
	yarn eslint src/ test/ --ext .ts,.tsx

test: build
	MOCHA_FILE=out/test-results.xml yarn c8 $(COVER_OPTS) mocha $(MOCHA_OPTS) "out/**/Test*.js"

# image-building targets
image:
	podman build -t docker-push.artifacts.apextoaster.com/ssube/conan-discord:main -f Containerfile .

image-local: ci
	podman pull docker-push.artifacts.apextoaster.com/ssube/conan-discord:main
	$(MAKE) image
	podman push docker-push.artifacts.apextoaster.com/ssube/conan-discord:main

# run targets
run: build
	node out/src/index.js
