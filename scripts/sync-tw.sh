#!/usr/bin/env bash
#
# sync-tw.sh — 自动生成繁体中文 README
#
# 用法：./scripts/sync-tw.sh
#
# 依赖：opencc（brew install opencc）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_ROOT/README.md"
DST="$REPO_ROOT/README.zh-TW.md"

if [[ ! -f "$SRC" ]]; then
  echo "[ERR] 未找到 README.md，请先创建 $SRC"
  exit 1
fi

if ! command -v opencc >/dev/null 2>&1; then
  echo "[ERR] 未找到 opencc，请先安装：brew install opencc"
  exit 1
fi

# 1. OpenCC 简体转台湾繁体
opencc -i "$SRC" -o "$DST" -c s2twp

# 2. 修正术语："智慧体"→"智能体"
sed -i '' 's/智慧體/智能體/g' "$DST"

# 3. 修正标题，加上"繁体中文"标记
sed -i '' 's/^# AgentHub.*/# AgentHub — AI 算法专家智能體團隊 繁體中文/' "$DST"

echo "[OK] 已生成 $DST"
