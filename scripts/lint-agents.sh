#!/usr/bin/env bash
#
# lint-agents.sh — 验证 AgentHub 智能体 Markdown 文件格式
#
# 检查规则：
#   ERROR: YAML frontmatter 必须包含 name、description（color 业务层必须，技术层可选）
#   WARN:  缺少推荐的 section 标题
#   WARN:  正文内容过少
#
# 用法: ./scripts/lint-agents.sh [--domain <domain>] [file ...]
#
#   --domain technical  仅检查技术层智能体
#   --domain business   仅检查业务层智能体
#   --domain all        检查所有（默认）

set -euo pipefail

# --- 路径 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TECH_DIRS=(
  ab-testing
  anomaly-detection
  clustering-classification
  cv
  fraud-detection
  graph
  nlp-text
  personalization
  recommendation
  reinforcement-learning
  search-ranking
  time-series
)

BUSINESS_DIRS=(
  business
)

# 技术层：name + description；业务层：name + description + color
TECH_REQUIRED=("name" "description")
BUSINESS_REQUIRED=("name" "description" "color")

# 推荐 section 模式（中英文）
RECOMMENDED_SECTIONS_PATTERNS=(
  "Identity|身份|记忆"
  "Core Mission|核心使命"
  "Critical Rules|关键规则"
)

errors=0
warnings=0

lint_file() {
  local file="$1"
  local dir
  dir="$(basename "$(dirname "$file")")"

  # 确定必需的 frontmatter 字段
  local required_fields=()
  if [[ "$dir" == "business" ]]; then
    required_fields=("${BUSINESS_REQUIRED[@]}")
  else
    required_fields=("${TECH_REQUIRED[@]}")
  fi

  # 1. 检查 frontmatter 分隔符
  local first_line
  first_line=$(head -1 "$file")
  if [[ "$first_line" != "---" ]]; then
    echo "ERROR $file: 缺少 frontmatter 开头 ---"
    errors=$((errors + 1))
    return
  fi

  # 提取 frontmatter
  local frontmatter
  frontmatter=$(awk 'NR==1{next} /^---$/{exit} {print}' "$file")

  if [[ -z "$frontmatter" ]]; then
    echo "ERROR $file: frontmatter 为空或格式错误"
    errors=$((errors + 1))
    return
  fi

  # 2. 检查必需的 frontmatter 字段
  local field
  for field in "${required_fields[@]}"; do
    if ! echo "$frontmatter" | grep -qE "^${field}:"; then
      echo "ERROR $file: 缺少 frontmatter 字段 '${field}' (${dir}/)"
      errors=$((errors + 1))
    fi
  done

  # 3. 检查推荐的 section（仅警告）
  local body
  body=$(awk 'BEGIN{n=0} /^---$/{n++; next} n>=2{print}' "$file")

  for pattern in "${RECOMMENDED_SECTIONS_PATTERNS[@]}"; do
    if ! echo "$body" | grep -qiE "$pattern"; then
      echo "WARN  $file: 缺少推荐 section (匹配: $pattern)"
      warnings=$((warnings + 1))
    fi
  done

  # 4. 检查文件是否有实质性内容
  if [[ $(echo "$body" | wc -w) -lt 50 ]]; then
    echo "WARN  $file: 正文内容过少 (< 50 词)"
    warnings=$((warnings + 1))
  fi
}

# 收集待检查文件
files=()
DOMAIN="${DOMAIN:-all}"

if [[ $# -gt 0 ]]; then
  # 命令行指定了文件
  files=("$@")
else
  # 自动扫描目录
  all_dirs=()
  if [[ "$DOMAIN" == "all" ]]; then
    all_dirs=("${TECH_DIRS[@]}" "${BUSINESS_DIRS[@]}")
  elif [[ "$DOMAIN" == "technical" ]]; then
    all_dirs=("${TECH_DIRS[@]}")
  elif [[ "$DOMAIN" == "business" ]]; then
    all_dirs=("${BUSINESS_DIRS[@]}")
  fi

  for dir in "${all_dirs[@]}"; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
      while IFS= read -r f; do
        files+=("$f")
      done < <(find "$REPO_ROOT/$dir" -name "*.md" -type f | sort)
    fi
  done
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "未找到智能体文件。"
  exit 1
fi

echo "正在检查 ${#files[@]} 个智能体文件..."
echo ""

for file in "${files[@]}"; do
  lint_file "$file"
done

echo ""
echo "结果: ${errors} 个错误, ${warnings} 个警告 (共 ${#files[@]} 个文件)"

if [[ $errors -gt 0 ]]; then
  echo "未通过: 请修复以上错误。"
  exit 1
else
  echo "已通过"
  exit 0
fi
