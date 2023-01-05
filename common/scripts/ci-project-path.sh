#! /bin/sh

PROJECT="${ROOT_PATH}"
PARENT="$(dirname "${PROJECT}")"
PROJECT_PATH="$(basename "${PARENT}")/$(basename "${PROJECT}")"

echo "${PROJECT_PATH}"