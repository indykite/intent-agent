# Intent Agent

[![Build Python project for master](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml/badge.svg)](https://github.com/indykite/intent-agent/actions/workflows/master-build.yaml)
[![Build & publish release artifacts](https://github.com/indykite/intent-agent/actions/workflows/publish-release.yaml/badge.svg)](https://github.com/indykite/intent-agent/actions/workflows/publish-release.yaml)
[![codecov](https://codecov.io/gh/indykite/intent-agent/branch/master/graph/badge.svg?token=gpcjoHjGyj)](https://codecov.io/gh/indykite/intent-agent) <!-- skipcq: SCT-1000 -->

## Release strategy

- `dev`: automatically on each push to `master`
- `rc`: automatically on each release
- `prod`: on each release via PR (needs to be merged manually)

## Local installation

These instructions are written for macOS 15, but mostly are OS-agnostic.

Pre-requisites:

- Python 3.13 (`brew install python@3.13`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`)
- GNU make `> 3.81` (`brew install make` on macOS)
    - **Warning: version that is included with macOS by default is too old (`GNU Make 3.81`)**
    - Check your version by running `/usr/bin/make --version`
    - If `make` is installed via Homebrew, it will be available as `gmake`. Make sure to use `gmake` instead of `make` in all the commands below.
    - Alternatively, set an alias or add the proper path to the `PATH`
        - for instance, add the following line to `~/.zshrc`: `export PATH="$HOMEBREW_PREFIX/opt/make/libexec/gnubin:$PATH"`

```sh
make install_dev
```

## Build/execution make targets

Although python execution is done through `uv`, all commands are defined in the `Makefile`.

| Command                          | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| `make -i`                        | Run all checks (doesn't change files)                 |
| `make fix`                       | Re-format files: imports, code style (Changes files)  |
| `make test`                      | Run unit tests                                        |
| `make test_coverage`             | Run code coverage and integration tests               |
| `make test_coverage_report_text` | Report code coverage as text in cli                   |
| `make test_coverage_report_html` | Report code coverage as an index.html file in htmlcov |

## Configuration

Example `.env` file can be copied from `.env.example`.
