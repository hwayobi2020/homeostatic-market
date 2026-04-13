# Chat Logs

이 대화창의 raw JSONL 기록을 보관하는 폴더.

## 파일

- `session_2026-04-10_to_11_homeostatic_phase6-8.jsonl` (22M)
  - 세션 ID: 98517603-7436-4629-9f31-5cac2e5040de
  - 기간: 2026-04-10 ~ 2026-04-11
  - 주요 내용: Phase 6~8 실험
    - 레버리지 (0~200%) 실험
    - MoE (Expert1/2/3 + Router) 실험
    - 데이터 v25 (52주 고저 피쳐) 생성
    - 피쳐 중요도 분석 (permutation)
    - LGBM 예측력 검증 (AUC 0.5 발견)
    - 분기/반기 rebalancing 실험
    - 평가 지표 재정립 (Calmar, AUC, Shortfall 등)
  - 결론: 원래 thesis "예측 지능 출현" 기각, Phase 9로 진화론적 관점 전환

## Raw 포맷

Claude Code의 대화 로그는 JSONL(JSON Lines). 각 줄이 하나의 메시지.

파싱 예:
```python
import json
with open('session_xxx.jsonl') as f:
    for line in f:
        msg = json.loads(line)
        print(msg.get('role'), msg.get('content'))
```
