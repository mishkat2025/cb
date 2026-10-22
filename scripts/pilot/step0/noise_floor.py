import sys, numpy as np, pandas as pd
j = pd.read_csv(sys.argv[1]); j["expert"] = j.pid.str[1:].astype(int) <= 24
g = j[j.expert].groupby(["ex", "pid", "var"]).avg_round
st = pd.DataFrame({"n": g.size(), "v": g.var(ddof=0), "rng": g.max() - g.min()}).reset_index()
st = st[st.n >= 2]
for ex, d in st.groupby("ex"):
    floor = np.sqrt((d["v"] * d.n).sum() / d.n.sum())
    print(f"{ex}: expert multi-view groups {len(d)}  mean range {d.rng.mean():.1f}  RMSE floor (score vs own-group mean) {floor:.2f}")

