#!/usr/bin/env bash
set -eu
SHDIR="$(readlink -f "$(dirname "${0}")")"
PYTHONPATH="${SHDIR}/../src" pylint --rcfile "${SHDIR}/../.pylintrc" "${@}"
