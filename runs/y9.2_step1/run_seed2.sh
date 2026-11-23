#!/bin/bash
# y9.2 1단계 seed 2: ver8 사전학습+fine-tune → Frangi 캐시 확장 → y9.2b → y9.2a
set -e
cd "/c/Users/annan/OneDrive/바탕 화면/umm"
export PYTHONIOENCODING=utf-8
python -W ignore wafixmatch.py run --method ver8 --seed 2 > runs/log_ver8_p0.05_s2.txt 2>&1
python -W ignore -c "
import os, numpy as np
from wafer import frangi as Fr
C = os.path.expanduser('~/wafer_cache')
ex = np.concatenate([np.load(os.path.join(C, f'ver3/runs/resnet18/p0.05/seed{k}/ssl_subset_random.npy')) for k in (0, 1, 2)])
Fr.build_labeled_cache(C, 'LSWMD.pkl', extra=ex)
" > runs/y9.2_step1/build_cache_s2.log 2>&1
for m in y9.2b y9.2a; do
  python -W ignore wafixmatch.py run --method $m --seed 2 > runs/log_${m}_p0.05_s2.txt 2>&1
done
echo ALL_DONE
