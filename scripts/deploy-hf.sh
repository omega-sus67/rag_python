#!/usr/bin/env bash
#
# Publish this repo to a Hugging Face Space.
#
# Why a script rather than a second git remote: a Space needs YAML frontmatter at
# the top of README.md, and putting that in the project README would leave a
# config table sitting above the title on GitHub. This keeps the two READMEs
# separate — GitHub gets the real one, the Space gets deploy/huggingface/README.md
# — without maintaining a divergent branch by hand.
#
# Usage:
#   scripts/deploy-hf.sh <space-git-url>
#   scripts/deploy-hf.sh https://huggingface.co/spaces/<user>/<space>
#
# Authentication: HF wants a write token as the password (your account password
# will not work). Create one at https://huggingface.co/settings/tokens and either
# let git prompt you, or embed it:
#   https://<user>:<hf_token>@huggingface.co/spaces/<user>/<space>

set -euo pipefail

SPACE_URL="${1:-}"
if [ -z "$SPACE_URL" ]; then
    echo "usage: $0 <space-git-url>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_README="$REPO_ROOT/deploy/huggingface/README.md"

if [ ! -f "$SPACE_README" ]; then
    echo "error: $SPACE_README is missing — it carries the Space frontmatter." >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "[deploy-hf] cloning the Space..."
git clone --quiet --depth 1 "$SPACE_URL" "$WORKDIR/space" 2>/dev/null || {
    echo "[deploy-hf] empty or unreachable Space; initialising a fresh repo."
    mkdir -p "$WORKDIR/space" && git -C "$WORKDIR/space" init --quiet
    git -C "$WORKDIR/space" remote add origin "$SPACE_URL"
}

# Ship the tracked working tree only. This is what keeps .env, the venv, the
# local corpora and every other gitignored file out of a public Space — the
# Space is a public git repo, so anything copied in is published.
echo "[deploy-hf] exporting tracked files..."
find "$WORKDIR/space" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$WORKDIR/space"

# Swap in the Space README, which carries the frontmatter HF requires.
cp "$SPACE_README" "$WORKDIR/space/README.md"

cd "$WORKDIR/space"
git add -A
if git diff --cached --quiet; then
    echo "[deploy-hf] nothing changed since the last publish."
    exit 0
fi

git -c user.email="deploy@local" -c user.name="deploy-hf" \
    commit --quiet -m "Deploy $(git -C "$REPO_ROOT" rev-parse --short HEAD)"

echo "[deploy-hf] pushing to the Space (this triggers a rebuild)..."
git push --quiet origin HEAD:main

echo "[deploy-hf] done. Watch the build log in the Space's 'Logs' tab."
