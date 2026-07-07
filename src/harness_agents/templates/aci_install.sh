#!/usr/bin/env bash
# Self-contained ACI (Agent-Computer Interface) — a SWE-agent-style windowed file
# editor as standalone /usr/local/bin commands. Pure bash + coreutils (no python/
# flake8/registry). State persists in $ACI_STATE across commands.
set -e
BIN="${ACI_BIN:-/usr/local/bin}"
STATE="${ACI_STATE:-$HOME/.aci_state}"
WINDOW="${ACI_WINDOW:-100}"
mkdir -p "$BIN"

# shared helpers sourced by every command
cat > "$BIN/_aci_common" <<COMMON
ACI_STATE="${STATE}"
ACI_WINDOW=${WINDOW}
_aci_get(){ grep -E "^\$1=" "\$ACI_STATE" 2>/dev/null | head -1 | cut -d= -f2-; }
_aci_set(){ # key val
  touch "\$ACI_STATE"; grep -vE "^\$1=" "\$ACI_STATE" > "\$ACI_STATE.tmp" 2>/dev/null || true
  echo "\$1=\$2" >> "\$ACI_STATE.tmp"; mv "\$ACI_STATE.tmp" "\$ACI_STATE"; }
_aci_print_window(){
  local cf=\$(_aci_get CURRENT_FILE) fl=\$(_aci_get FIRST_LINE)
  [ -z "\$cf" ] && { echo "No file open. Use: open <path>"; return 1; }
  [ -z "\$fl" ] && fl=1
  local n=\$(wc -l < "\$cf"); n=\${n:-0}
  local end=\$(( fl + ACI_WINDOW - 1 )); [ \$end -gt \$n ] && end=\$n
  [ \$fl -lt 1 ] && fl=1
  echo "[File: \$cf (\$n lines total)]"
  [ \$fl -gt 1 ] && echo "(\$(( fl - 1 )) more lines above)"
  sed -n "\${fl},\${end}p" "\$cf" | nl -ba -v "\$fl"
  [ \$end -lt \$n ] && echo "(\$(( n - end )) more lines below)"
}
COMMON

# open <path> [line]
cat > "$BIN/open" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
[ -z "$1" ] && { _aci_print_window; exit 0; }
f="$1"; [ "${f:0:1}" != "/" ] && f="$(pwd)/$f"
[ -f "$f" ] || { echo "File not found: $1"; exit 1; }
_aci_set CURRENT_FILE "$f"
ln="${2:-1}"; w=$ACI_WINDOW; fl=$(( ln - w/2 )); [ $fl -lt 1 ] && fl=1
_aci_set FIRST_LINE "$fl"
_aci_print_window
EOF

# goto <line>
cat > "$BIN/goto" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
[ -z "$1" ] && { echo "Usage: goto <line>"; exit 1; }
w=$ACI_WINDOW; fl=$(( $1 - w/2 )); [ $fl -lt 1 ] && fl=1
_aci_set FIRST_LINE "$fl"; _aci_print_window
EOF

# scroll_down / scroll_up
cat > "$BIN/scroll_down" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
fl=$(_aci_get FIRST_LINE); fl=${fl:-1}; _aci_set FIRST_LINE $(( fl + ACI_WINDOW - 2 )); _aci_print_window
EOF
cat > "$BIN/scroll_up" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
fl=$(_aci_get FIRST_LINE); fl=${fl:-1}; nf=$(( fl - ACI_WINDOW + 2 )); [ $nf -lt 1 ] && nf=1; _aci_set FIRST_LINE $nf; _aci_print_window
EOF

# create <file>
cat > "$BIN/create" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
[ -z "$1" ] && { echo "Usage: create <file>"; exit 1; }
f="$1"; [ "${f:0:1}" != "/" ] && f="$(pwd)/$f"
touch "$f"; _aci_set CURRENT_FILE "$f"; _aci_set FIRST_LINE 1; _aci_print_window
EOF

# edit <start> <end>   (replacement text read from stdin until EOF)
cat > "$BIN/edit" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
cf=$(_aci_get CURRENT_FILE); [ -z "$cf" ] && { echo "No file open. Use: open <path>"; exit 1; }
s="$1"; e="$2"
case "$s" in ''|*[!0-9]*) echo "Usage: edit <start_line> <end_line>  (then the new text, ended by EOF)"; exit 1;; esac
case "$e" in ''|*[!0-9]*) echo "Usage: edit <start_line> <end_line>"; exit 1;; esac
new=$(mktemp); cat > "$new"
tmp=$(mktemp); n=$(wc -l < "$cf")
[ $s -gt 1 ] && sed -n "1,$(( s - 1 ))p" "$cf" > "$tmp"
cat "$new" >> "$tmp"
[ $e -lt $n ] && sed -n "$(( e + 1 )),${n}p" "$cf" >> "$tmp"
cp "$tmp" "$cf"; rm -f "$tmp" "$new"
echo "[File edited: lines $s-$e replaced]"
_aci_set FIRST_LINE $(( s - ACI_WINDOW/2 > 0 ? s - ACI_WINDOW/2 : 1 ))
_aci_print_window
EOF

# search_dir <query> [dir]
cat > "$BIN/search_dir" <<'EOF'
#!/usr/bin/env bash
[ -z "$1" ] && { echo "Usage: search_dir <query> [dir]"; exit 1; }
d="${2:-.}"; grep -rniI --include='*.py' -e "$1" "$d" 2>/dev/null | head -100
EOF
# search_file <query> [file]
cat > "$BIN/search_file" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/_aci_common"
[ -z "$1" ] && { echo "Usage: search_file <query> [file]"; exit 1; }
f="${2:-$(_aci_get CURRENT_FILE)}"; [ -z "$f" ] && { echo "No file open/given"; exit 1; }
grep -nI -e "$1" "$f" 2>/dev/null | head -100
EOF
# find_file <name> [dir]
cat > "$BIN/find_file" <<'EOF'
#!/usr/bin/env bash
[ -z "$1" ] && { echo "Usage: find_file <name> [dir]"; exit 1; }
find "${2:-.}" -name "$1" 2>/dev/null | head -100
EOF

chmod +x "$BIN/open" "$BIN/goto" "$BIN/scroll_down" "$BIN/scroll_up" "$BIN/create" "$BIN/edit" "$BIN/search_dir" "$BIN/search_file" "$BIN/find_file"
echo "ACI installed to $BIN (state: $STATE, window: $WINDOW)"
