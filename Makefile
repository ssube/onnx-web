export GIT_BRANCH ?= $(shell git rev-parse --abbrev-ref HEAD)
export GIT_COMMIT ?= $(shell git rev-parse HEAD)
export GIT_OPTIONS ?=
export GIT_REMOTES ?= $(shell git remote -v | awk '{ print $1; }' | sort | uniq)
export GIT_TAG ?= $(shell git tag -l --points-at HEAD | head -1)

.PHONY: git-stats git-push

# from https://gist.github.com/amitchhajer/4461043#gistcomment-2349917
git-stats: ## print git contributor line counts (approx, for fun)
	git ls-files | while read f; do git blame -w -M -C -C --line-porcelain "$$f" |\
		grep -I '^author '; done | sort -f | uniq -ic | sort -n

git-push: ## push to both github and gitlab
	git push $(GIT_ARGS) github $(GIT_BRANCH)
	git push $(GIT_ARGS) gitlab $(GIT_BRANCH)
