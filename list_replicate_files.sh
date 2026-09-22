#!/usr/bin/env bash
# Print every regular file below the current directory.
# Run this after changing into the replicate directory.

set -euo pipefail

find . -type f -print
