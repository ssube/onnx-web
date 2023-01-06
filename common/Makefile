# Paths
# resolve the makefile's path and directory, from https://stackoverflow.com/a/18137056
export MAKE_PATH    ?= $(abspath $(firstword $(MAKEFILE_LIST)))
export ROOT_PATH    ?= $(dir $(MAKE_PATH))
export SCRIPT_PATH	?= $(ROOT_PATH)/common/scripts

# git
export GIT_BRANCH ?= $(shell git rev-parse --abbrev-ref HEAD)
export GIT_COMMIT ?= $(shell git rev-parse HEAD)
export GIT_OPTIONS ?=
export GIT_REMOTES ?= $(shell git remote -v | awk '{ print $1; }' | sort | uniq)
export GIT_TAG ?= $(shell git tag -l --points-at HEAD | head -1)

# CI
export CI_COMMIT_REF_SLUG ?= $(GIT_BRANCH)
export CI_COMMIT_SHA ?= $(GIT_COMMIT)
export CI_COMMIT_TAG ?= $(GIT_TAG)
export CI_ENVIRONMENT_SLUG ?= local
export CI_JOB_ID ?= 0
export CI_PROJECT_PATH ?= $(shell ROOT_PATH=$(ROOT_PATH) ${SCRIPT_PATH}/ci-project-path.sh)
export CI_RUNNER_DESCRIPTION ?= $(shell hostname)
export CI_RUNNER_ID ?= $(shell hostname)
export CI_RUNNER_VERSION ?= 0.0.0

.PHONY: git-stats git-push help

# from https://gist.github.com/amitchhajer/4461043#gistcomment-2349917
git-stats: ## print git contributor line counts (approx, for fun)
	git ls-files | while read f; do git blame -w -M -C -C --line-porcelain "$$f" |\
		grep -I '^author '; done | sort -f | uniq -ic | sort -n

git-push: ## push to both github and gitlab
	git push $(GIT_ARGS) github $(GIT_BRANCH)
	git push $(GIT_ARGS) gitlab $(GIT_BRANCH)

# from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## print this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort \
		| sed 's/^.*\/\(.*\)/\1/' \
		| awk 'BEGIN {FS = ":[^:]*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

release:
	$$(yarn global bin)/commit-and-tag-version --packageFiles gui/package.json
