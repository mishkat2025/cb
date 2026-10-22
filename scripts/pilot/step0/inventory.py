"""Step 0 inventory: parse every MobiPhysio filename and tabulate counts."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])

# E01_P01_AF_VFL_GM.mp4  (separator tolerant: '_', ' ', '-' or none)
PAT = re.compile(
    r"^(?P<ex>E\d{2})[ _-]*(?P<pid>P\d{1,3})[ _-]*A(?P<angle>[FLR])[ _-]*"
    r"V(?P<var>FL|ML|LL|LJ|HJ|O|LR)[ _-]*G(?P<gender>[MF])$",
    re.IGNORECASE,
)

rows, fails = [], []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r"E\d{2}", p.name)):
    for f in sorted(d.glob("*.mp4")):
        m = PAT.match(f.stem.strip())
        if not m:
            fails.append(str(f))
            continue
        g = {k: v.upper() for k, v in m.groupdict().items()}
        g["folder"] = d.name
        g["file"] = f.name
        g["size_mb"] = f.stat().st_size / 1e6
        rows.append(g)

total = len(rows) + len(fails)
print(f"files: {total}  parsed: {len(rows)}  failed: {len(fails)}")
for f in fails:
    print("  FAIL:", f)

# folder/exercise consistency
bad_folder = [r for r in rows if not r["folder"].upper().startswith(r["ex"])]
print("exercise code != folder prefix:", len(bad_folder))

for ex in sorted({r["ex"] for r in rows}):
    R = [r for r in rows if r["ex"] == ex]
    print(f"\n===== {ex}: {len(R)} videos")
    print(" angle    :", dict(sorted(Counter(r["angle"] for r in R).items())))
    print(" variation:", dict(sorted(Counter(r["var"] for r in R).items())))
    print(" gender   :", dict(sorted(Counter(r["gender"] for r in R).items())))
    pids = sorted({r["pid"] for r in R}, key=lambda s: int(s[1:]))
    print(f" participants ({len(pids)}):", " ".join(pids))
    # gender consistency per participant id
    pg = defaultdict(set)
    for r in R:
        pg[r["pid"]].add(r["gender"])
    mixed = {p: g for p, g in pg.items() if len(g) > 1}
    print(" pids with >1 gender:", mixed or "none")
    per_pid = Counter(r["pid"] for r in R)
    print(" videos per pid:", dict(sorted(Counter(per_pid.values()).items())), "(count->#pids)")
    # per pid: variation x angle matrix summary
    sig = Counter()
    for p in pids:
        vs = defaultdict(set)
        for r in R:
            if r["pid"] == p:
                vs[r["var"]].add(r["angle"])
        sig[" ".join(f"{v}:{''.join(sorted(a))}" for v, a in sorted(vs.items()))] += 1
    print(" per-pid (variation:angles) patterns:")
    for s, c in sig.most_common():
        print(f"   {c:3d} x  {s}")
    # multi-view groups
    grp = defaultdict(set)
    for r in R:
        grp[(r["pid"], r["var"])].add(r["angle"])
    sizes = Counter("".join(sorted(a)) for a in grp.values())
    print(f" (pid,variation) groups: {len(grp)}  view-sets:", dict(sizes))
    full = sum(1 for a in grp.values() if a == {"F", "L", "R"})
    print(f" complete F+L+R groups: {full}")
    sz = [r["size_mb"] for r in R]
    print(f" size MB: min {min(sz):.1f}  median {sorted(sz)[len(sz)//2]:.1f}  max {max(sz):.1f}")

# cross-exercise participant overlap (same P-number + gender)
exs = sorted({r["ex"] for r in rows})
if len(exs) > 1:
    keys = {ex: {(r["pid"], r["gender"]) for r in rows if r["ex"] == ex} for ex in exs}
    common = set.intersection(*keys.values())
    print(f"\n(pid,gender) present in all of {exs}: {len(common)}")
    for ex in exs:
        print(f"  only in {ex}:", sorted(keys[ex] - common, key=lambda k: int(k[0][1:])))

import csv
out = Path(sys.argv[2])
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote", out)
