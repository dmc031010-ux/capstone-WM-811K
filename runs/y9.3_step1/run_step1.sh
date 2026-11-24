#!/bin/bash
# y9.3 1단계: 캐시 (length, scaled) → 점검 → seed 0·1·2 × y9.3a, y9.3b
set -e
cd "/c/Users/annan/OneDrive/바탕 화면/umm"
export PYTHONIOENCODING=utf-8
python -W ignore -c "
import os
from wafer import frangi as Fr
Fr.build_y93_cache(os.path.expanduser('~/wafer_cache'), 'LSWMD.pkl')
" > runs/y9.3_step1/build_cache.log 2>&1
python -W ignore runs/y9.3_step1/check_cache.py > runs/y9.3_step1/check_cache.txt 2>&1
for s in 0 1 2; do
  for m in y9.3a y9.3b; do
    python -W ignore wafixmatch.py run --method $m --seed $s > runs/log_${m}_p0.05_s${s}.txt 2>&1
  done
done
echo ALL_DONE
