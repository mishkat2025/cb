"""Pick complete F+L+R groups for one exercise: n_pid participants x 2 variations (so that
same-person/different-variation pairs act as unsynchronized controls).
usage: select_groups_ex.py <inventory.csv> <out.txt> <EX> <n_pid> <max_pnum> [seed]"""
import os
import random
import sys
from pathlib import Path

import pandas as pd
# dataset root: env MOBIPHYSIO_ROOT, else the folder that contains the cvdc-physio repo
DATA_ROOT = Path(os.environ.get("MOBIPHYSIO_ROOT", Path(__file__).resolve().parents[4] / "dataset" / "videos"))

inv = pd.read_csv(sys.argv[1])
out, ex, n_pid, max_p = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
inv = inv[inv.ex == ex].copy()
inv["pnum"] = inv.pid.str[1:].astype(int)
full = inv.groupby(["pid", "var"]).angle.apply(lambda a: set(a) == {"F", "L", "R"})
full = full[full].reset_index()[["pid", "var"]]
full = full[full.pid.str[1:].astype(int) <= max_p]
picked = []
for pid in rng.sample(sorted(full.pid.unique()), n_pid):
    for v in rng.sample(sorted(full[full.pid == pid]["var"]), 2):
        picked.append((pid, v))
sel = pd.DataFrame(picked, columns=["pid", "var"])
print(sel.to_string(index=False))
files = inv.merge(sel, on=["pid", "var"])
out.write_text("\n".join(str(DATA_ROOT / f / n) for f, n in zip(files.folder, files.file)), encoding="utf-8")
print(len(sel), "groups,", len(files), "videos ->", out)
