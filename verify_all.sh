#!/usr/bin/env bash
# Run every runnable rung and report PASS/FAIL. CPU-only; ~80s once models are cached.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export HF_HUB_DISABLE_XET=1
FAIL=0
for r in salmon rung1_features_and_circuits rung2_residual_stream rung3_superposition \
         rung4_induction_heads rung5_sparse_autoencoders rung7_attribution_graphs; do
  START=$(date +%s)
  if (cd "$r" && $PY starter.py >/tmp/spinup_$r.out 2>/tmp/spinup_$r.err); then
    printf "  PASS  %-32s %3ds\n" "$r" "$(( $(date +%s) - START ))"
  else
    printf "  FAIL  %-32s %s\n" "$r" "$(tail -2 /tmp/spinup_$r.err | head -1)"
    FAIL=1
  fi
done
# --- tool gates: our recommendations held to the standard we hold papers to ---
# These test an EXTERNAL line of evidence against a signal a rung already measured, rather than
# re-running our own script and agreeing with ourselves. Skipped if optional deps are absent.
for g in tools/gate_autointerp.py; do
  n=$(basename "$g" .py)
  START=$(date +%s)
  $PY "$g" >/tmp/spinup_$n.out 2>/tmp/spinup_$n.err; rc=$?
  case $rc in
    0) printf "  PASS  %-32s %3ds  (gate)\n" "$n" "$(( $(date +%s) - START ))" ;;
    2) printf "  SKIP  %-32s optional dep missing (pip install datasets)\n" "$n" ;;
    *) printf "  FAIL  %-32s %s\n" "$n" "$(grep -m1 . /tmp/spinup_$n.out | tail -1)"; FAIL=1 ;;
  esac
done

[ $FAIL -eq 0 ] && echo "All rungs ran. Output kept in /tmp/spinup_*.out" \
               || echo "Some rungs failed -- see PITFALLS.md"
exit $FAIL
