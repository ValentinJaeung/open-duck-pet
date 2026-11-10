#!/usr/bin/env bash
# 로컬 LLM 서버(Ollama)를 켜고 모델을 받습니다. 사용자 폴더(~/.local/ollama)에 설치된 것을 씁니다(sudo 불필요).
# 사용법: bash scripts/run_ollama.sh [모델명]     기본 qwen3:8b (약 5GB, 도구 호출 지원)
# 이 터미널은 서버가 계속 돌므로 켜 둔 채로 두고, 다른 터미널에서 작업하세요.
set -euo pipefail
MODEL="${1:-qwen3:8b}"
OLLAMA="$HOME/.local/ollama/bin/ollama"
export LD_LIBRARY_PATH="$HOME/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
if ! curl -s -m 2 localhost:11434/api/version >/dev/null; then
  "$OLLAMA" serve > "$HOME/.local/ollama/serve.log" 2>&1 &
  for i in $(seq 1 30); do curl -s -m 1 localhost:11434/api/version >/dev/null && break; sleep 1; done
fi
echo "Ollama $(curl -s localhost:11434/api/version)"
"$OLLAMA" pull "$MODEL"
echo "준비 완료: $MODEL  (GPU 사용량: nvidia-smi)"
wait
