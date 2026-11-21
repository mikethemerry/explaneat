# WHAS500 at the standard budget (exploratory, 21 Nov 2026)

Mike's question: what happens to the WHAS500 Cox case at the population and generations of the
other studies (population 250, 100 generations) instead of the survival config's 150 / 60?
Only those two settings change; everything else is the Cox case unchanged (same split file,
`configs/survival.ini`, identity output, Efron Cox loss, fitness -loss - k/D, 5 epochs per
generation, phantom mask, selection on the validation fold). Not used in the thesis unless Mike
decides to switch; the presented network stays `../runs/whas500-s0`.

    H=experiments/survival/cox_case
    for s in 0 1 2 3 4; do
      PYTHONHASHSEED=0 OMP_NUM_THREADS=1 uv run --extra survival python experiments/survival/run_seed.py \
        whas500 $s --pop-size 250 --generations 100 --split-file $H/splits/whas500.splits.json \
        --test-size 0.2 --val-size 0.2 --split-seed 42 --horizons 365,1095,1825 \
        --calibration-horizon 1095 --out $H/standard_budget/runs --tag whas500-s$s
    done

Outputs: `runs/whas500-s<k>.{json,genome.pkl,log}` (same record format as `../runs`).
