"""IPL career stats for an identified player, from data/deliveries.csv."""
from functools import lru_cache

import pandas as pd

from .config import DELIVERIES_CSV, STATS_NAME_MAP


@lru_cache(maxsize=1)
def _df():
    return pd.read_csv(DELIVERIES_CSV)


def player_stats(folder_name: str):
    name = STATS_NAME_MAP.get(folder_name)
    if name is None:
        return None
    d = _df()
    bat = d[d.batsman == name]
    runs = int(bat.batsman_runs.sum())
    balls = int((bat.wide_runs == 0).sum())
    bowl = d[d.bowler == name]
    not_bowler = ["run out", "retired hurt", "obstructing the field"]
    wk = bowl[bowl.dismissal_kind.notna() & ~bowl.dismissal_kind.isin(not_bowler)]
    return {
        "IPL name": name,
        "Runs": runs,
        "Balls faced": balls,
        "Strike rate": round(100 * runs / balls, 1) if balls else None,
        "Fours": int((bat.batsman_runs == 4).sum()),
        "Sixes": int((bat.batsman_runs == 6).sum()),
        "Dismissals": int((d.player_dismissed == name).sum()),
        "Wickets": int(len(wk)),
    }
