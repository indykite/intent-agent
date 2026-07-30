.ONESHELL:

.PHONY: build fix codestyle_check clean install install_dev upgrade_dev project_check \
	fix codestyle_check test test_coverage test_coverage_report_text \
	test_coverage_report_html test_coverage_report_xml start_pipeline integration \
	build_integration build dry_run build_docker_image run_docker_image

build_integration: build integration dry_run
build: project_check codestyle_check test
dry_run:


-include .env

clean:
	@set -e
	@rm -rf dist
	@rm -rf htmlcov
	@rm -f .coverage
	@rm -rf .venv

install:
	@set -e
	# this would fail if uv.lock is outdated
	uv lock --check

install_dev:
	# this would fail if uv.lock is outdated
	uv lock --check

upgrade_dev:
	uv lock --upgrade

project_check:
	@set -e
	uv lock --check
	# fails loudly if .venv has drifted from uv.lock, instead of surfacing as a
	# confusing import error later in codestyle_check/test
	uv sync --locked
	uv run --locked validate-pyproject pyproject.toml

fix:
	uv run --locked ruff format
	uv run --locked ruff check --fix

codestyle_check:
	uv run --locked ruff check

test:
	uv run --locked pytest test

integration:
	uv run --locked pytest integration

# This target contains combined coverage from unit & integration tests
test_coverage:
	@set -e
	uv run --locked pytest --cov=. test
	uv run --locked pytest --cov=. --cov-append integration

test_coverage_report_text:
	@uv run --locked coverage report -i

test_coverage_report_html:
	@uv run --locked coverage html -i

test_coverage_report_xml:
	@uv run --locked coverage xml -i

build_docker_image:
	docker buildx build --platform linux/amd64 . \
		-t europe-west1-docker.pkg.dev/indykite-mgmt/indykite/intent-agent:local

run_docker_image:
	docker run --platform=linux/amd64 \
		europe-west1-docker.pkg.dev/indykite-mgmt/indykite/intent-agent:local
