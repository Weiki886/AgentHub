#!/usr/bin/env bash
#
# convert.sh — 将智能体 .md 文件转换为各工具专用格式（AgentHub 中文算法专家版）
#
# 读取 AgentHub 目录中的 .md 文件，输出到 integrations/<tool>/。
# 添加或修改智能体后运行此脚本重新生成集成文件。
#
# 用法：
#   ./scripts/convert.sh [--tool <name>] [--domain <domain>] [--out <dir>] [--help]
#
# 支持的工具：
#   antigravity  — Antigravity skill 文件 (~/.gemini/antigravity/skills/)
#   gemini-cli   — Gemini CLI 扩展 (skills/ + gemini-extension.json)
#   opencode     — OpenCode agent 文件 (.opencode/agent/*.md)
#   cursor       — Cursor rule 文件 (.cursor/rules/*.mdc)
#   aider        — 单文件 CONVENTIONS.md for Aider
#   windsurf     — 单文件 .windsurfrules for Windsurf
#   openclaw     — OpenClaw SOUL.md 文件 (openclaw_workspace/<agent>/SOUL.md)
#   qwen         — Qwen Code SubAgent 文件 (~/.qwen/agents/*.md)
#   all          — 所有工具（默认）
#
# 支持的领域：
#   technical    — 仅技术层智能体（算法工程师）
#   business     — 仅业务层智能体（业务分析师）
#   all          — 所有智能体（默认）
#
# 输出到仓库根目录下的 integrations/<tool>/。
# 此脚本不会修改用户配置目录 — 参见 install.sh。

set -euo pipefail

# --- 颜色辅助 ---
if [[ -t 1 ]]; then
  GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; BOLD=''; RESET=''
fi

info()    { printf "${GREEN}[OK]${RESET}  %s\n" "$*"; }
warn()    { printf "${YELLOW}[!!]${RESET}  %s\n" "$*"; }
error()   { printf "${RED}[ERR]${RESET} %s\n" "$*" >&2; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# --- 路径 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/integrations"
TODAY="$(date +%Y-%m-%d)"

# 技术层智能体目录
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

# 业务层智能体目录
BUSINESS_DIRS=(
  business
)

# --- 用法 ---
usage() {
  sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# --- Frontmatter 辅助函数 ---

# 从 YAML frontmatter 中提取单个字段值
get_field() {
  local field="$1" file="$2"
  awk -v f="$field" '
    /^---$/ { fm++; next }
    fm == 1 && $0 ~ "^" f ": " { sub("^" f ": ", ""); print; exit }
  ' "$file"
}

# 去除 frontmatter，返回正文部分
get_body() {
  awk 'BEGIN{fm=0} /^---$/{fm++; next} fm>=2{print}' "$1"
}

# 从文件名生成 slug（直接用文件名，business 目录加 business- 前缀）
slugify_from_file() {
  local file="$1"
  local dir="$(basename "$(dirname "$file")")"
  local base="$(basename "$file" .md)"
  if [[ "$dir" == "business" ]]; then
    echo "business-${base}"
  else
    echo "$base"
  fi
}

# --- 颜色映射 ---
resolve_opencode_color() {
  local c="$1"
  case "$c" in
    cyan)           echo "#00FFFF" ;;
    blue)           echo "#3498DB" ;;
    green)          echo "#2ECC71" ;;
    red)            echo "#E74C3C" ;;
    purple)         echo "#9B59B6" ;;
    orange)         echo "#F39C12" ;;
    teal)           echo "#008080" ;;
    indigo)         echo "#6366F1" ;;
    pink)           echo "#E84393" ;;
    gold)           echo "#EAB308" ;;
    amber)          echo "#F59E0B" ;;
    neon-green)     echo "#10B981" ;;
    neon-cyan)      echo "#06B6D4" ;;
    metallic-blue)  echo "#3B82F6" ;;
    yellow)         echo "#EAB308" ;;
    violet)         echo "#8B5CF6" ;;
    rose)           echo "#F43F5E" ;;
    lime)           echo "#84CC16" ;;
    gray)           echo "#6B7280" ;;
    fuchsia)        echo "#D946EF" ;;
    *)              echo "$c" ;;
  esac
}

# --- 各工具转换器 ---

convert_antigravity() {
  local file="$1"
  local name description slug outdir outfile body

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outdir="$OUT_DIR/antigravity/$slug"
  outfile="$outdir/SKILL.md"
  mkdir -p "$outdir"

  cat > "$outfile" <<HEREDOC
---
name: ${slug}
description: ${description}
risk: low
source: community
date_added: '${TODAY}'
---
${body}
HEREDOC
}

convert_gemini_cli() {
  local file="$1"
  local name description slug outdir outfile body

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outdir="$OUT_DIR/gemini-cli/skills/$slug"
  outfile="$outdir/SKILL.md"
  mkdir -p "$outdir"

  cat > "$outfile" <<HEREDOC
---
name: ${slug}
description: ${description}
---
${body}
HEREDOC
}

convert_opencode() {
  local file="$1"
  local name description color slug outfile body

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  local raw_color
  raw_color="$(get_field "color" "$file" | tr -d '"')"
  color="$(resolve_opencode_color "$raw_color")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outfile="$OUT_DIR/opencode/agents/${slug}.md"
  mkdir -p "$OUT_DIR/opencode/agents"

  cat > "$outfile" <<HEREDOC
---
name: ${name}
description: ${description}
mode: subagent
color: "${color}"
---
${body}
HEREDOC
}

convert_cursor() {
  local file="$1"
  local name description slug outfile body

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outfile="$OUT_DIR/cursor/rules/${slug}.mdc"
  mkdir -p "$OUT_DIR/cursor/rules"

  cat > "$outfile" <<HEREDOC
---
description: ${description}
globs: ""
alwaysApply: false
---
${body}
HEREDOC
}

convert_openclaw() {
  local file="$1"
  local name description slug outdir body
  local soul_content="" agents_content=""

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outdir="$OUT_DIR/openclaw/$slug"
  mkdir -p "$outdir"

  # 按 ## 标题关键词拆分为 SOUL.md（人设）和 AGENTS.md（业务）
  local current_target="agents"
  local current_section=""

  while IFS= read -r line; do
    if [[ "$line" =~ ^##[[:space:]] ]]; then
      if [[ -n "$current_section" ]]; then
        if [[ "$current_target" == "soul" ]]; then
          soul_content+="$current_section"
        else
          agents_content+="$current_section"
        fi
      fi
      current_section=""

      local header_lower
      header_lower="$(echo "$line" | tr '[:upper:]' '[:lower:]')"

      if [[ "$header_lower" =~ identity ]] ||
         [[ "$header_lower" =~ 身份 ]] ||
         [[ "$header_lower" =~ 记忆 ]] ||
         [[ "$header_lower" =~ communication ]] ||
         [[ "$header_lower" =~ 沟通 ]] ||
         [[ "$header_lower" =~ style ]] ||
         [[ "$header_lower" =~ 风格 ]] ||
         [[ "$header_lower" =~ critical.rule ]] ||
         [[ "$header_lower" =~ 关键规则 ]] ||
         [[ "$header_lower" =~ rules.you.must.follow ]]; then
        current_target="soul"
      else
        current_target="agents"
      fi
    fi

    current_section+="$line"$'\n'
  done <<< "$body"

  if [[ -n "$current_section" ]]; then
    if [[ "$current_target" == "soul" ]]; then
      soul_content+="$current_section"
    else
      agents_content+="$current_section"
    fi
  fi

  cat > "$outdir/SOUL.md" <<HEREDOC
${soul_content}
HEREDOC

  cat > "$outdir/AGENTS.md" <<HEREDOC
${agents_content}
HEREDOC

  cat > "$outdir/IDENTITY.md" <<HEREDOC
# ${name}
${description}
HEREDOC
}

convert_qwen() {
  local file="$1"
  local name description tools slug outfile body

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  tools="$(get_field "tools" "$file")"
  slug="$(slugify_from_file "$file")"
  body="$(get_body "$file")"

  outfile="$OUT_DIR/qwen/agents/${slug}.md"
  mkdir -p "$(dirname "$outfile")"

  if [[ -n "$tools" ]]; then
    cat > "$outfile" <<HEREDOC
---
name: ${slug}
description: ${description}
tools: ${tools}
---
${body}
HEREDOC
  else
    cat > "$outfile" <<HEREDOC
---
name: ${slug}
description: ${description}
---
${body}
HEREDOC
  fi
}

# Aider 和 Windsurf 是单文件格式，先累积再统一写入
AIDER_TMP="$(mktemp)"
WINDSURF_TMP="$(mktemp)"
trap 'rm -f "$AIDER_TMP" "$WINDSURF_TMP"' EXIT

cat > "$AIDER_TMP" <<'HEREDOC'
# AgentHub — AI 算法专家智能体团队
#
# 本文件为 Aider 提供完整的 AI 算法专家智能体阵容。
# 来源：https://github.com/Weiki886/AgentHub
#
# 激活方式：在 Aider 会话中引用智能体名称，例如：
#   "使用推荐系统工程师智能体帮我设计推荐策略"
#
# 由 scripts/convert.sh 生成 — 请勿手动编辑。

HEREDOC

cat > "$WINDSURF_TMP" <<'HEREDOC'
# AgentHub — AI 算法专家智能体团队
#
# 完整的 AI 算法专家智能体阵容（技术层 + 业务层）。
# 激活方式：在 Windsurf 对话中引用智能体名称。
#
# 由 scripts/convert.sh 生成 — 请勿手动编辑。

HEREDOC

accumulate_aider() {
  local file="$1"
  local name description body dir

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  body="$(get_body "$file")"
  dir="$(basename "$(dirname "$file")")"

  local tag="[技术]"
  [[ "$dir" == "business" ]] && tag="[业务]"

  cat >> "$AIDER_TMP" <<HEREDOC

---

## ${name}

${tag} ${description}

${body}
HEREDOC
}

accumulate_windsurf() {
  local file="$1"
  local name description body dir

  name="$(get_field "name" "$file")"
  description="$(get_field "description" "$file")"
  body="$(get_body "$file")"
  dir="$(basename "$(dirname "$file")")"

  local tag="[技术]"
  [[ "$dir" == "business" ]] && tag="[业务]"

  cat >> "$WINDSURF_TMP" <<HEREDOC

================================================================================
## ${name}  ${tag}
${description}
================================================================================

${body}

HEREDOC
}

# --- 主循环 ---

run_conversions() {
  local tool="$1"
  local count=0

  # 确定要扫描的目录列表
  local all_dirs=()
  if [[ "$DOMAIN" == "all" ]]; then
    all_dirs=("${TECH_DIRS[@]}" "${BUSINESS_DIRS[@]}")
  elif [[ "$DOMAIN" == "technical" ]]; then
    all_dirs=("${TECH_DIRS[@]}")
  elif [[ "$DOMAIN" == "business" ]]; then
    all_dirs=("${BUSINESS_DIRS[@]}")
  fi

  for dir in "${all_dirs[@]}"; do
    local dirpath="$REPO_ROOT/$dir"
    [[ -d "$dirpath" ]] || continue

    while IFS= read -r -d '' file; do
      local first_line
      first_line="$(head -1 "$file")"
      [[ "$first_line" == "---" ]] || continue

      local name
      name="$(get_field "name" "$file")"
      [[ -n "$name" ]] || continue

      case "$tool" in
        antigravity) convert_antigravity "$file" ;;
        gemini-cli)  convert_gemini_cli  "$file" ;;
        opencode)    convert_opencode    "$file" ;;
        cursor)      convert_cursor      "$file" ;;
        openclaw)    convert_openclaw    "$file" ;;
        qwen)        convert_qwen        "$file" ;;
        aider)       accumulate_aider    "$file" ;;
        windsurf)    accumulate_windsurf "$file" ;;
      esac

      (( count++ )) || true
    done < <(find "$dirpath" -name "*.md" -type f -print0 | sort -z)
  done

  echo "$count"
}

# --- 入口 ---

main() {
  local tool="all"
  DOMAIN="all"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tool)   tool="${2:?'--tool 需要一个值'}"; shift 2 ;;
      --domain) DOMAIN="${2:?'--domain 需要一个值'}"; shift 2 ;;
      --out)    OUT_DIR="${2:?'--out 需要一个值'}"; shift 2 ;;
      --help|-h) usage ;;
      *) error "未知选项: $1"; usage ;;
    esac
  done

  local valid_tools=("antigravity" "gemini-cli" "opencode" "cursor" "aider" "windsurf" "openclaw" "qwen" "all")
  local valid=false
  for t in "${valid_tools[@]}"; do [[ "$t" == "$tool" ]] && valid=true && break; done
  if ! $valid; then
    error "未知工具 '$tool'。可选: ${valid_tools[*]}"
    exit 1
  fi

  local valid_domains=("technical" "business" "all")
  local domain_valid=false
  for d in "${valid_domains[@]}"; do [[ "$d" == "$DOMAIN" ]] && domain_valid=true && break; done
  if ! $domain_valid; then
    error "未知领域 '$DOMAIN'。可选: ${valid_domains[*]}"
    exit 1
  fi

  header "AgentHub — 转换为工具专用格式"
  echo "  仓库:   $REPO_ROOT"
  echo "  输出:   $OUT_DIR"
  echo "  工具:   $tool"
  echo "  领域:   $DOMAIN"
  echo "  日期:   $TODAY"

  local tools_to_run=()
  if [[ "$tool" == "all" ]]; then
    tools_to_run=("antigravity" "gemini-cli" "opencode" "cursor" "aider" "windsurf" "openclaw" "qwen")
  else
    tools_to_run=("$tool")
  fi

  local total=0
  for t in "${tools_to_run[@]}"; do
    header "正在转换: $t"
    local count
    count="$(run_conversions "$t")"
    total=$(( total + count ))

    if [[ "$t" == "gemini-cli" ]]; then
      mkdir -p "$OUT_DIR/gemini-cli"
      cat > "$OUT_DIR/gemini-cli/gemini-extension.json" <<'HEREDOC'
{
  "name": "agenthub",
  "version": "1.0.0",
  "description": "AgentHub — AI 算法专家智能体团队",
  "source": "https://github.com/Weiki886/AgentHub"
}
HEREDOC
      info "已写入 gemini-extension.json"
    fi

    info "已转换 $count 个智能体 ($t)"
  done

  if [[ "$tool" == "all" || "$tool" == "aider" ]]; then
    mkdir -p "$OUT_DIR/aider"
    cp "$AIDER_TMP" "$OUT_DIR/aider/CONVENTIONS.md"
    info "已写入 integrations/aider/CONVENTIONS.md"
  fi
  if [[ "$tool" == "all" || "$tool" == "windsurf" ]]; then
    mkdir -p "$OUT_DIR/windsurf"
    cp "$WINDSURF_TMP" "$OUT_DIR/windsurf/.windsurfrules"
    info "已写入 integrations/windsurf/.windsurfrules"
  fi

  echo ""
  info "完成。共转换: $total"
}

main "$@"
