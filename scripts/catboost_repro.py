"""Run one month fold of a recipe twice with one seed and print the max absolute probability gap.

python scripts/catboost_repro.py recipes/catboost.json [month] [seed]
"""

import sys
import time

import numpy as np

from ringtrace import store
from ringtrace.cascade import fit_predict
from ringtrace.run import cascade_spec, feature_matrix, load_recipe

recipe = load_recipe(sys.argv[1])
month = int(sys.argv[2]) if len(sys.argv) > 2 else 9
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
spec = cascade_spec(recipe)
spec.seeds = (seed,)
led = store.ledger()
X, _ = feature_matrix(recipe)
period = led["period"].to_numpy()
y = led["y"].to_numpy().astype(np.int64)
tr = np.flatnonzero(~led["is_test"].to_numpy())
fit, hold = tr[period[tr] != month], tr[period[tr] == month]
runs = []
for _ in range(2):
    t0 = time.time()
    runs.append(fit_predict(X[fit], y[fit], period[fit], [X[hold]], spec)[0])
    print(f"run done in {time.time() - t0:.0f}s", flush=True)
gap = np.abs(runs[0] - runs[1])
print(
    f"month {month} seed {seed}: max abs diff {gap.max():.3e}, mean {gap.mean():.3e}, "
    f"argmax changes {int((runs[0].argmax(1) != runs[1].argmax(1)).sum())} of {len(hold)}"
)
