#!/usr/bin/env bash
# run_all.sh 로 켠 노드들을 모두 끕니다. (Ollama 서버는 건드리지 않음)
# 이 스크립트 자신과 그 상위 프로세스는 제외합니다(pkill -f 는 자기 자신을 죽일 수 있음).
PAT='duck_sim_bridge\.bridge_node|duck_perception\.perception_node|duck_behavior\.behavior_node|install/duck_[a-z]+/lib/duck_[a-z]+/|scripts/run_(bridge|skills|perception|brain|behavior)\.sh'
skip=" $$ "; p=$$
while [ "$p" -gt 1 ]; do p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break; skip="$skip$p "; done
for pid in $(ps -eo pid,args | grep -E "$PAT" | grep -v grep | awk '{print $1}'); do
  case "$skip" in *" $pid "*) continue;; esac
  kill "$pid" 2>/dev/null || true
done
sleep 1
left=0
for pid in $(ps -eo pid,args | grep -E "$PAT" | grep -v grep | awk '{print $1}'); do case "$skip" in *" $pid "*) ;; *) left=$((left+1));; esac; done
echo "정리 후 남은 노드 프로세스: $left"
