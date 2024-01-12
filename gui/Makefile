.PHONY: build bundle ci clean docs docs-local lint package run test

# JS targets
node_modules: deps

ci: deps lint build-shebang test bundle

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
	cp -v src/components/main.css out/src/components/

build-shebang: build
	sed -i '1s;^;#! /usr/bin/env node\n\n;g' $(shell pwd)/out/src/main.js
	chmod ug+x out/src/main.js

bundle: build
	node esbuild.js
# copy other files into output to make a complete UI
	cp -v src/index.html out/
	cp -v src/config.json out/
# copy everything into the server's default path
	cp -v src/index.html ../api/gui/
	cp -v src/config.json ../api/gui/
	cp -v out/bundle/main.css ../api/gui/bundle/
	cp -v out/bundle/main.js ../api/gui/bundle/

COVER_OPTS := --all \
		--exclude ".eslintrc.js" \
		--exclude "docs/**" \
    --exclude "out/bundle/**" \
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

watch: deps
	WATCH=TRUE make ci
