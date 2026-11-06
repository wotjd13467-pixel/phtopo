#!/usr/bin/env bash
# run_pn.sh -- Simphony(pn.x) 실행 드라이버
#
# 사용법 (WSL, build_inputs.py 를 돌린 폴더에서):
#   bash run_pn.sh band_loto                 # ./band_loto/pn.in 실행
#   bash run_pn.sh band_loto findnodes_17    # 여러 개 차례로
#   bash run_pn.sh -h                        # 이 도움말
#
# 하는 일: ./<case>/ 로 들어가서
#   - pn.in 의 Hrfile 을 ./hr/ 에서 심볼릭 링크
#   - pn.x 실행, 표준출력을 pn.log 로 저장 (결과 PN.out, bulkek.dat, Nodes.dat ... 는 그 폴더에)
#   - 종료코드/소요시간을 STATUS 파일에 기록
# pn.x 위치는 환경변수 PNX (기본 ~/simphony/bin/pn.x)

set -u
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
PNX="${PNX:-$HOME/simphony/bin/pn.x}"
[ -x "$PNX" ] || { echo "pn.x not found/executable at $PNX (PNX=... 로 지정)" >&2; exit 1; }

for CASE in "$@"; do
    d="$PWD/$CASE"
    [ -f "$d/pn.in" ] || { echo "SKIP $CASE (no pn.in)"; continue; }

    hr=$(sed -n "s/.*Hrfile *= *'\([^']*\)'.*/\1/p" "$d/pn.in" | head -1)
    ln -sf "../hr/$hr" "$d/$hr"

    printf '%-16s ... ' "$CASE"
    t0=$(date +%s)
    ( cd "$d" && "$PNX" > pn.log 2>&1 )
    rc=$?
    t1=$(date +%s)
    printf 'rc=%d  %ds\n' "$rc" "$((t1 - t0))"
    { echo "case      : $CASE"
      echo "pn.x      : $PNX"
      echo "elapsed_s : $((t1 - t0))"
      echo "exit_code : $rc"
    } > "$d/STATUS"
done
