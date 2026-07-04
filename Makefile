.PHONY: build publish publish-test

build:
	rm -rf dist && uv build

publish: build
	. ~/.secrets/pypi && uv publish && \
	v=v$$(uv version --short) && git tag $$v && git push origin $$v

# ponytail: timestamp as devN — unique every run, no counter state to track
publish-test:
	@base=$$(uv version --short); \
	uv version $$base.dev$$(date +%s) && \
	rm -rf dist && uv build && \
	{ . ~/.secrets/test_pypi && uv publish --publish-url https://test.pypi.org/legacy/; }; \
	uv version $$base
