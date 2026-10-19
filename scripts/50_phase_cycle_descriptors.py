"""Descriptive statistics of inlet-phase timing quoted in Sections 3.1 and 3.3 and SI S1."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("P3F_ROOT", Path(__file__).resolve().parents[1]))


def main() -> None:
    x = pd.read_parquet(ROOT / "data" / "processed" / "forecast_10min_features.parquet",
                        columns=["PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET", "transition_inlet_recent_1h", "split"])
    inlet = x["PROCESSPHASE_INLET"].round()
    change = inlet.ne(inlet.shift())
    group = change.cumsum()
    runs = inlet.groupby(group).size() / 6.0  # phase durations in hours
    starts = x.index[change & inlet.eq(1)]
    cycles = np.diff(starts.values).astype("timedelta64[m]").astype(float)
    size = group.map(group.value_counts())
    remaining_h = (size - x.groupby(group).cumcount() - 1) / 6.0
    stable_test = x["split"].eq("test") & x["transition_inlet_recent_1h"].eq(0)
    out = {
        "inlet_phase_duration_min": {"median": float(runs.median() * 60), "q25": float(runs.quantile(0.25) * 60), "q75": float(runs.quantile(0.75) * 60)},
        "inlet_cycle_min": {"median": float(np.median(cycles)), "q25": float(np.percentile(cycles, 25)), "q75": float(np.percentile(cycles, 75))},
        "no_transition_test_windows": int(stable_test.sum()),
        "share_phase_ending_within_30min": float((remaining_h[stable_test] <= 0.5).mean()),
        "non_integer_inlet_codes": int((x["PROCESSPHASE_INLET"] % 1 != 0).sum()),
        "non_integer_outlet_codes": int((x["PROCESSPHASE_OUTLET"] % 1 != 0).sum()),
    }
    (ROOT / "tables" / "Phase_cycle_descriptors.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
