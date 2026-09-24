.ONESHELL:

.PHONY: build fix codestyle_check clean install install_dev upgrade_dev project_check \
	fix codestyle_check test test_coverage test_coverage_report_text \
	test_coverage_report_html test_coverage_report_xml start_pipeline integration \
	build_integration build dry_run build_docker_image start_docker_app start_app \
	start_docker_compose stop_docker_compose destroy_docker_compose \
	start_mock_introspect_service evals

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

evals:
	@set -e
	mkdir -p tmp/evals
	uv run --locked python -m evals --markdown tmp/evals/report.md --json tmp/evals/report.json

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
	docker buildx build --load --platform linux/amd64,linux/arm64 . \
		-t europe-west1-docker.pkg.dev/indykite-mgmt/indykite/intent-agent:local

start_docker_app:
	docker run \
		-p 6003:6003 -p 8001:8001 --rm --env-file .env \
		europe-west1-docker.pkg.dev/indykite-mgmt/indykite/intent-agent:local

start_app:
	uv run --locked python3 -m indykite.intent_agent

start_docker_compose:
	@set -e
	docker compose build
	docker compose up -d --wait

stop_docker_compose:
	docker compose stop

destroy_docker_compose:
	docker compose down --remove-orphans

start_mock_introspect_service:
	uv run --locked python3 -m manual_testing.mock_introspect
