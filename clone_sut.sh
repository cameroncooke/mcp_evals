#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT_DIR="${ROOT}/sut"
REPO_DIR="${SUT_DIR}/HackerNews"
REPO_URL="https://github.com/emergetools/HackerNews"

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Repo already exists at ${REPO_DIR}"
  exit 0
fi

mkdir -p "${SUT_DIR}"
git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
echo "Cloned ${REPO_URL} to ${REPO_DIR}"
