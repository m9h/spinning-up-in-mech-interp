#!/usr/bin/env bash
# Run every runnable rung and report PASS/FAIL. CPU-only; ~80s once models are cached.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export HF_HUB_DISABLE_XET=1
FAIL=0
for r in rung1_features_and_circuits rung2_residual_stream rung3_superposition \
         rung4_induction_heads rung5_sparse_autoencoders rung7_attribution_graphs; do
  START=$(date +%s)
  if (cd "$r" && $PY starter.py >/tmp/spinup_$r.out 2>/tmp/spinup_$r.err); then
    printf "  PASS  %-32s %3ds\n" "$r" "$(( $(date +%s) - START ))"
  else
    printf "  FAIL  %-32s %s\n" "$r" "$(tail -2 /tmp/spinup_$r.err | head -1)"
    FAIL=1
  fi
done
[ $FAIL -eq 0 ] && echo "All rungs ran. Output kept in /tmp/spinup_*.out" \
               || echo "Some rungs failed -- see PITFALLS.md"
exit $FAIL
