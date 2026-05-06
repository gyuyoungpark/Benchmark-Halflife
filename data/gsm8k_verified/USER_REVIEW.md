# GSM8K 30-item Spot-Check — User Review (2차 annotator)

**현재 상태:** Claude (1차 annotator)가 30 items 모두 평가 완료. 결과 → `llm_annotator_30.json`

| Label | Count |
|---|---|
| equivalent (3/3 yes) | 29 |
| partial (2/3 yes) | 1 (idx 56) |
| non-equivalent | 0 |

## 사용자가 할 일 (선택 옵션)

### 옵션 A — 전체 검토 (~30분, 가장 정직)
30 items 전체를 직접 한번씩 보고 본인 평가를 기록. Cohen's κ 계산 가능.

### 옵션 B — 핵심만 검토 (~10분, 권장)
다음 5 items만 직접 확인:

1. **idx 56** (Bob/Alice — 1차에서 "partial" 판정됨)
   - orig: 정수 차원 (2×8×2=32, /2=160 bags)
   - pert: 소수 차원 (3×9×1.5=40.5, /3=162 bags)
   - 사용자 판단: 이 차이가 "non-equivalent"인지 "partial이라도 acceptable"인지 결정
   
2. **idx 47** (Jerry/Megan algebra)
   - 본 논문 abstract 첫 문장 anecdote의 원본 item
   - 정말 elementary arithmetic만으로 풀리는지 확인

3. **idx 57** (Bill bottles vs Samantha juice)
   - orig은 "70+70=140" (additive doubling), pert는 "50*3=150" (multiplication)
   - 같은 op type으로 봐야 하는가? 1차 annotator는 "equivalent"로 판정했지만 reviewer가 다르게 볼 수도

4. **idx 43** (Movies/Books) — 8 ops, 가장 복잡한 item
   - 모든 step이 정말 parallel한지

5. **idx 102** (Tow truck/Delivery)
   - "first three days" → "first four days" (자연어 변경 + 숫자 변경 동시) — 둘이 정말 같은 구조인지

### 옵션 C — 신뢰 기반 빠른 결정 (~3분)
1차 annotator 결과를 신뢰. idx 56만 "partial"로 처리하고 나머지 29를 "equivalent"로 인정.

## 결과 기록 양식

본인 평가를 마치면 다음과 같이 한 줄 요약을 알려주세요:

```
"옵션 B 검토 완료. 5 items 중 [N]개가 equivalent로 동의, [M]개는 partial/non-equivalent. 
1차와 disagreement는 idx [...]. 따라서 30 items 전체 estimate: equivalent [X], partial [Y], non-equivalent [Z]."
```

또는 옵션 C라면:

```
"옵션 C: 1차 annotator 결과 신뢰. equivalent 29, partial 1."
```

이 결과를 바탕으로 §4에 한 문장 추가:

> "On a 30-item random subsample of the verified set, two annotators (one author, one Claude-Sonnet-4-6) independently rated equivalence on three criteria; **N**/30 items were rated fully equivalent by both."

## 1차 annotator의 단일 partial 사례 (idx 56)

```
ORIG: Bob, 10 raised beds 2×8×2 ft each = 32 cu ft per bed.
       Total 320 cu ft; bags hold 2 cu ft → 160 bags × $12 = $1,920.
       (모든 숫자 정수)

PERT: Alice, 12 flower beds 3×9×1.5 ft each = 40.5 cu ft per bed.
       Total 486 cu ft; bags hold 3 cu ft → 162 bags × $15 = $2,430.
       (1.5 ft 차원 → 40.5 소수 등장)
```

**Reviewer 시각**: "operation count 같지만 decimal 곱셈이 추가되어 arithmetic이 미세하게 더 어려움."

**우리 시각**: "step structure는 동일. 1.5는 단순한 0.5 단위라 carry/borrow 복잡도는 비슷."

사용자 결정:
- (a) "이 정도면 equivalent" → equivalent: 30 (모두)
- (b) "1차의 partial 판정이 맞다" → equivalent: 29, partial: 1 ✓ (default)
- (c) "이건 non-equivalent" → equivalent: 29, non-equivalent: 1

## 이후 본문 적용

사용자 결정 알려주시면 §4 fidelity caveat에 한 문장 추가하고 PDF 재컴파일하겠습니다.
