"""Community-QBR Pipeline (v0.1) - nflreadpy / nflfastR + FTN"""
import os
import numpy as np
import pandas as pd
import nflreadpy as nfl

# ---------- Konfiguration ----------
import datetime
_today = datetime.date.today()
CURRENT = _today.year if _today.month >= 9 else _today.year - 1
SEASONS = list(range(2022, CURRENT + 1))  # inkl. laufender Saison
START = 375
CLUTCH_WP = 0.40      # Clutch-x2 unter dieser Win Probability (nflfastR wp)
BIGPLAY_WPA = 0.05    # Big-Play-Bonus ab diesem wpa
FG_LINE = 35          # yardline_100 <= 35 entspricht ca. 52-Yard-FG
SACK = -20
INCOMPLETION = -10
GRADES = [(775, "A+"), (642, "A"), (575, "A-"), (525, "B+"), (450, "B"),
          (425, "B-"), (400, "C+"), (375, "C"), (350, "C-"), (300, "D+"),
          (250, "D"), (200, "D-"), (125, "F+"), (75, "F"), (0, "F-")]


def grade(points):
    for limit, g in GRADES:
        if points >= limit:
            return g
    return "F-"


def yard_pts(y):
    """Raumgewinn-Skala (positiv und negativ)."""
    return np.select(
        [y >= 20, y >= 10, y >= 1, y <= -20, y <= -10, y <= -1],
        [25, 10, 5, -25, -10, -5], default=0)


def load():
    pbp = nfl.load_pbp(SEASONS).to_pandas()
    print("PBP-Saisons:", sorted(pbp.season.unique()))
    # FTN-Charting erscheint fuer die laufende Saison oft verzoegert -> pro Saison laden
    parts = []
    for s in SEASONS:
        try:
            parts.append(nfl.load_ftn_charting([s]).to_pandas())
        except Exception as e:
            print(f"FTN {s} nicht verfuegbar ({e}), ohne FTN-Werte weiter")
    ftn = pd.concat(parts, ignore_index=True)
    ftn = ftn[["nflverse_game_id", "nflverse_play_id",
               "is_qb_out_of_pocket", "is_catchable_ball"]]
    return pbp.merge(ftn, how="left",
                     left_on=["game_id", "play_id"],
                     right_on=["nflverse_game_id", "nflverse_play_id"])


def score_plays(df):
    # --- Filter: nur Pass/Run, keine Fouls, Spikes, Kneels, 2-Pt ---
    df = df[df.play_type.isin(["pass", "run"])]
    df = df[(df.penalty != 1) & (df.qb_spike != 1) & (df.qb_kneel != 1)
            & (df.two_point_attempt != 1)].copy()

    # --- QB-Zuordnung ---
    qb_ids = set(df.passer_player_id.dropna())
    is_pass = df.qb_dropback.eq(1) & ~df.qb_scramble.eq(1)
    is_scr = df.qb_scramble.eq(1)
    is_designed = df.play_type.eq("run") & df.rusher_player_id.isin(qb_ids) & ~is_scr
    df["qb_id"] = np.where(is_pass, df.passer_player_id,
                  np.where(is_scr | is_designed,
                           df.rusher_player_id.fillna(df.passer_player_id), None))
    df["qb_name"] = np.where(is_pass, df.passer_player_name,
                    np.where(is_scr | is_designed,
                             df.rusher_player_name.fillna(df.passer_player_name), None))
    df = df[df.qb_id.notna()].copy()
    is_pass = df.qb_dropback.eq(1) & ~df.qb_scramble.eq(1)
    is_sack = df.sack.eq(1)
    is_run = ~is_pass & ~is_sack
    is_comp = df.complete_pass.eq(1)

    # --- Raumgewinn (QB-Credit: Air Yards bei Pass, Yards bei Lauf) ---
    credited = np.where(is_run | is_sack, df.yards_gained, np.where(is_comp, df.air_yards, 0))
    credited = pd.Series(credited, index=df.index).fillna(0)
    yards_pts = yard_pts(credited)  # Sacks stapeln: Negativ-Raum + Sack-Wert
    # Screens/Check-downs mit <=0 Air Yards: keine Wertung (0)
    yards_pts = np.where(is_comp & (credited <= 0), 0, yards_pts)
    # Incompletions: kein Raumgewinn
    yards_pts = np.where(is_pass & ~is_comp & ~is_sack, 0, yards_pts)

    td = (df.pass_touchdown.eq(1) | df.rush_touchdown.eq(1)) * 50
    first = (df.first_down_pass.eq(1) | df.first_down_rush.eq(1)) * 5

    # FG-Range: Linie wird durch den QB-Credit-Raumgewinn erstmals ueberschritten
    end_line = df.yardline_100 - credited
    fg = ((df.yardline_100 > FG_LINE) & (end_line <= FG_LINE) & ~is_sack) * 15

    on_run = ((is_comp & df.is_qb_out_of_pocket.eq(1)) * 5)
    inacc = ((is_pass & (df.is_catchable_ball == 0)) * -15)  # FTN: nicht fangbar

    incomp = (is_pass & ~is_comp & ~is_sack) * INCOMPLETION
    sack = is_sack * SACK
    failed = ((df.down == 3) & ~(df.first_down_pass.eq(1) | df.first_down_rush.eq(1)
              | df.pass_touchdown.eq(1) | df.rush_touchdown.eq(1))) * -15
    failed = failed + ((df.down == 4) & ~(df.first_down_pass.eq(1) | df.first_down_rush.eq(1)
              | df.pass_touchdown.eq(1) | df.rush_touchdown.eq(1))) * -20
    bigplay = (df.wpa >= BIGPLAY_WPA) * 50

    base = yards_pts + td + first + fg + on_run + inacc + incomp + sack + failed + bigplay
    df["mult"] = np.where(df.wp < CLUTCH_WP, 2, 1)
    df["pts"] = base * df.mult
    return df


def game_scores(df):
    g = (df.groupby(["season", "week", "game_id", "qb_id", "qb_name", "posteam"],
                    as_index=False)
           .agg(plays=("pts", "size"), pts=("pts", "sum")))
    g["score"] = START + g.pts
    g["grade"] = g.score.apply(grade)
    return g


if __name__ == "__main__":
    games = game_scores(score_plays(load()))
    # Nur Starter-Level-Spiele (>= 15 Plays)
    games = games[games.plays >= 15]
    print(games.score.describe())
    print(games.grade.value_counts(normalize=True).sort_index())
    os.makedirs("docs", exist_ok=True)
    out = games[["season", "week", "qb_name", "posteam", "plays", "score", "grade"]].copy()
    out["score"] = out.score.round(0).astype(int)
    out.to_json("docs/data.json", orient="records", force_ascii=False)
