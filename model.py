"""Own-model fantasy engine: context-weighted history + matchup + game environment + injuries."""
import datetime as dt
import json
import pathlib
import re
import sqlite3
import time

import numpy as np
import pandas as pd
import requests
import nflreadpy as nfl

SKILL = ["QB", "RB", "WR", "TE"]
ENV = {"QB": .9, "WR": .8, "TE": .6, "RB": .5, "K": .8}  # how much team scoring environment moves a position
INJ = {"questionable": .92, "doubtful": .25, "out": 0, "ir": 0, "pup": 0, "sus": 0}
BAD = ("out", "ir", "pup", "sus", "doubtful")
PA_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]  # edit to your league's D/ST rules
DST_ID = {"LA": "LAR"}  # nflverse -> Sleeper team ids
AVG, SHRINK = 22.5, 1.5
DB = "ffapp.db"
SLEEPER = "https://api.sleeper.app/v1"
KCOLS = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59",
         "fg_made_60_", "fg_missed", "pat_made", "pat_missed", "fg_att"]


# ---------- data ----------
def season_now():
    t = dt.date.today()
    return t.year if t.month >= 8 else t.year - 1


def _each(fn, yrs):
    out = []
    for y in yrs:
        try:
            out.append(fn([y]).to_pandas())
        except Exception:
            pass
    return pd.concat(out, ignore_index=True)


def load(season):
    yrs = [season - 2, season - 1, season]
    return _each(nfl.load_player_stats, yrs), _each(nfl.load_team_stats, yrs), _each(nfl.load_schedules, yrs)


def prep(ps):
    ps = ps[(ps.season_type == "REG") & ps.position.isin(SKILL + ["K"])].copy()
    ps[KCOLS] = ps[KCOLS].fillna(0)
    kick = (3 * (ps.fg_made_0_19 + ps.fg_made_20_29 + ps.fg_made_30_39) + 4 * ps.fg_made_40_49
            + 5 * (ps.fg_made_50_59 + ps.fg_made_60_) + ps.pat_made - ps.fg_missed - ps.pat_missed)
    ps["pts"] = np.where(ps.position == "K", kick, ps.fantasy_points_ppr.fillna(0))
    ps["t"] = ps.season * 100 + ps.week
    return ps


def team_games(sch):
    s = sch[sch.game_type == "REG"]

    def side(me, op, sg):
        return pd.DataFrame(dict(
            season=s.season, week=s.week, team=s[f"{me}_team"], opp=s[f"{op}_team"],
            implied=(s.total_line + sg * s.spread_line) / 2, opp_implied=(s.total_line - sg * s.spread_line) / 2,
            kick=pd.to_datetime(s.gameday.astype(str) + " " + s.gametime.fillna("13:00").astype(str), errors="coerce")
            .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT"),
            qb=s[f"{me}_qb_id"], coach=s[f"{me}_coach"], pf=s[f"{me}_score"], pa=s[f"{op}_score"], roof=s.roof))

    tg = pd.concat([side("home", "away", 1), side("away", "home", -1)], ignore_index=True)
    tg["t"] = tg.season * 100 + tg.week
    return tg


def next_week(sch, season):
    n = team_games(sch)
    n = n[(n.season == season) & n.pf.isna()]
    return int(n.week.min()) if len(n) else None


def finished(sch, ps):
    """(season, week, team) games that are final AND already in the stats feed."""
    tg = team_games(sch)
    tg = tg[tg.pf.notna()]
    have = set(zip(ps.season, ps.week, ps.team))
    return {k for k in zip(tg.season, tg.week, tg.team) if k in have}


def open_teams(sch, season, week):
    g = team_games(sch)
    g = g[(g.season == season) & (g.week == week)]
    return set(g[g.kick > pd.Timestamp.now(tz="America/New_York")].team)


def sleeper_players():
    f = pathlib.Path("sleeper_players.json")
    if f.exists() and time.time() - f.stat().st_mtime < 43200:
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    try:
        p = requests.get(f"{SLEEPER}/players/nfl", timeout=60).json()
        if p:
            f.write_text(json.dumps(p))
            return p
    except Exception:
        pass
    return json.loads(f.read_text()) if f.exists() else {}


def nk(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


def injuries(season, week, sp):
    m = {}
    try:
        inj = nfl.load_injuries([season]).to_pandas()
        for r in inj[inj.week == week].itertuples():
            if isinstance(r.report_status, str):
                m[r.gsis_id] = r.report_status.lower()
    except Exception:
        pass
    for p in sp.values():  # live Sleeper status wins
        if p.get("gsis_id") and p.get("injury_status"):
            m[p["gsis_id"]] = p["injury_status"].lower()
    return m


def rostered(league_id, sp):
    """-> (keys, n_found, error). keys match on gsis id, Sleeper id (D/ST) or name|pos."""
    try:
        teams = requests.get(f"{SLEEPER}/league/{league_id.strip()}/rosters", timeout=20).json()
    except Exception as e:
        return set(), 0, f"Couldn't reach Sleeper ({e.__class__.__name__})."
    if not isinstance(teams, list) or not teams:
        return set(), 0, "Sleeper didn't recognize that league ID. Use the number from sleeper.com/leagues/<ID>."
    ids = {i for t in teams for i in (t.get("players") or [])}
    keys = set(ids)
    for i in ids:
        p = sp.get(i) or {}
        keys.add(p.get("gsis_id"))
        pos = "DST" if p.get("position") == "DEF" else p.get("position")
        keys.add(nk(f"{p.get('first_name', '')}{p.get('last_name', '')}") + "|" + str(pos))
    return keys, len(ids), None


# ---------- projections ----------
def wts(n, same_qb, same_coach):
    w = np.maximum(.9 ** np.arange(n)[::-1], .15)  # recency, but old seasons never fully vanish
    if same_qb is not None:
        w = w * np.where(same_qb, 1, .35)  # games with a different QB count far less
    return w * np.where(same_coach, 1, .6)  # ...and so do games under a different play-caller


def run(ps, ts, sch, season, week, inj=None):
    inj = inj or {}
    tg = team_games(sch)
    now = season * 100 + week
    past = ps[ps.t < now].merge(tg[["season", "week", "team", "implied", "qb", "coach"]],
                                on=["season", "week", "team"], how="left")
    nxt = tg[tg.t == now].set_index("team")
    played = tg[(tg.t < now) & tg.pf.notna()].sort_values("t")
    last = played.groupby("team").tail(1).set_index("team")
    cut = played.groupby("team").t.apply(lambda s: s.nlargest(6).min())
    nm = ps.drop_duplicates("player_id", keep="last").set_index("player_id").player_display_name
    prior = past.groupby("position").pts.mean()

    alw = past.groupby(["opponent_team", "position", "t"]).pts.sum().reset_index().sort_values("t")
    lg = alw.groupby("position").pts.mean()
    rec = alw.groupby(["opponent_team", "position"]).tail(8).groupby(["opponent_team", "position"]).pts.agg(["mean", "count"])
    lgv = lg.reindex(rec.index.get_level_values(1)).values
    ratio = ((rec["mean"] * rec["count"] + 3 * lgv) / ((rec["count"] + 3) * lgv)).clip(.75, 1.3)

    def coach(team):
        c = nxt.coach.get(team)
        return c if isinstance(c, str) else last.coach.get(team)

    def cur_qb(team):
        q = last.qb.get(team)
        if inj.get(q) in BAD:  # starter hurt -> expect the backup who's been throwing
            alts = past[(past.team == team) & (past.position == "QB") & (past.player_id != q)
                        & (past.t >= cut.get(team, 0))].groupby("player_id").attempts.sum().sort_values()
            alts = [a for a in alts.index[::-1] if inj.get(a) not in BAD]
            if alts:
                return alts[0]
        return q

    rows = []
    for pid, g in past.groupby("player_id"):
        g = g.sort_values("t")
        r = g.iloc[-1]
        team, pos = r.team, r.position
        if team not in nxt.index or len(g) < 2 or r.t < cut.get(team, 0):
            continue
        gn = nxt.loc[team]
        qb, co = cur_qb(team), coach(team)
        sq = None if pos in ("QB", "K") else (g.qb == qb).values
        sc = (g.coach == co).values
        w, x = wts(len(g), sq, sc), g.pts.values
        base = (w @ x + SHRINK * prior[pos]) / (w.sum() + SHRINK)
        sd = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
        hi = np.average(g.implied.fillna(AVG), weights=w)
        impl = gn.implied if pd.notna(gn.implied) else hi
        env = base * ENV[pos] * (impl - hi) / AVG
        rt = ratio.get((gn.opp, pos), 1.0)
        mu = base * .5 * (rt - 1)
        dome = .4 if pos == "K" and gn.roof in ("dome", "closed") else 0
        st = inj.get(pid, "")
        mult = INJ.get(st, 1)
        proj = (base + env + mu + dome) * mult
        mt = sc if sq is None else (sc & sq)
        t4 = g.tail(4)
        use = {"WR": f"{t4.target_share.mean():.0%} target share (last 4)", "TE": f"{t4.target_share.mean():.0%} target share (last 4)",
               "RB": f"{(t4.carries + t4.targets).mean():.0f} touches+targets/game (last 4)",
               "QB": f"{t4.attempts.mean():.0f} pass att/game (last 4)", "K": f"{t4.fg_att.mean():.1f} FG att/game (last 4)"}[pos]
        why = [f"Baseline {base:.1f} pts from {len(g)} games, weighted toward same QB/play-caller",
               f"Game environment: team implied {impl:.1f} vs usual {hi:.1f} ({env:+.1f})",
               f"Matchup vs {gn.opp}: allows {rt:.0%} of avg to {pos} ({mu:+.1f})", f"Usage: {use}"]
        if dome:
            why.append("Dome/closed roof (+0.4)")
        if mult < 1:
            why.append(f"Injury report: {st} (x{mult})")
        if 0 < mt.sum() < len(g):
            why.append(f"{x[mt].mean():.1f} ppg in {mt.sum()} matching-context games vs {x[~mt].mean():.1f} in {(~mt).sum()} others")
        flags = []
        if sq is not None and sq.sum() < 6:
            flags.append(f"QB change: {nm.get(qb, '?')} ({sq.sum()} of {len(g)} games together)")
        if sc.sum() < 6:
            flags.append(f"New play-caller: {co}")
        rows.append(dict(player_id=pid, player=r.player_display_name, pos=pos, team=team, opp=gn.opp, proj=proj,
                         floor=max(proj - .8 * sd, 0), ceil=proj + .8 * sd, trend=t4.pts.tail(3).mean() - base,
                         status=st, mult=mult, why=why, flags=flags))

    # ---- D/ST ----
    d = ts[ts.season_type == "REG"].merge(tg[["season", "week", "team", "pa", "coach", "opp_implied"]],
                                          on=["season", "week", "team"], how="left")
    d["t"] = d.season * 100 + d.week
    d = d[(d.t < now) & d.pa.notna()].copy()
    tier = d.pa.map(lambda p: next(v for cap, v in PA_TIERS if p <= cap))
    d["pts"] = (d.def_sacks + 2 * d.def_interceptions + 2 * d.fumble_recovery_opp + 6 * d.def_tds
                + 2 * d.def_safeties + 2 * (d.def_fg_blocks + d.def_punt_blocks + d.def_pat_blocks) + tier)
    dlg = d.pts.mean()
    orec = d.sort_values("t").groupby("opponent_team").tail(8).groupby("opponent_team").pts.agg(["mean", "count"])
    dratio = ((orec["mean"] * orec["count"] + 3 * dlg) / ((orec["count"] + 3) * dlg)).clip(.75, 1.3)
    for team in nxt.index:
        g = d[d.team == team].sort_values("t")
        if len(g) < 2:
            continue
        gn, co = nxt.loc[team], coach(team)
        sc = (g.coach == co).values
        w, x = wts(len(g), None, sc), g.pts.values
        base = (w @ x + SHRINK * dlg) / (w.sum() + SHRINK)
        sd = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
        ho = np.average(g.opp_implied.fillna(AVG), weights=w)
        oi = gn.opp_implied if pd.notna(gn.opp_implied) else ho
        env = (ho - oi) * .3
        rt = dratio.get(gn.opp, 1.0)
        mu = base * .5 * (rt - 1)
        proj = base + env + mu
        why = [f"Baseline {base:.1f} D/ST pts from {len(g)} games",
               f"Opponent implied only {oi:.1f} vs usual {ho:.1f} faced ({env:+.1f})",
               f"{gn.opp} offense gives up {rt:.0%} of avg D/ST points ({mu:+.1f})"]
        rows.append(dict(player_id=DST_ID.get(team, team), player=f"{team} D/ST", pos="DST", team=team, opp=gn.opp,
                         proj=proj, floor=max(proj - .8 * sd, 0), ceil=proj + .8 * sd, trend=g.pts.tail(3).mean() - base,
                         status="", mult=1, why=why, flags=[] if sc.sum() >= 6 else [f"New play-caller: {co}"]))
    return pd.DataFrame(rows)


# ---------- snapshots + hit/miss grading ----------
def con():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS snaps(season INT, week INT, player_id TEXT, player TEXT, pos TEXT, team TEXT,
                 my_proj REAL, my_rank INT, cons_proj REAL, cons_rank INT, actual REAL,
                 PRIMARY KEY(season, week, player_id))""")
    return c


def consensus(season, week, sp):
    pos = "&".join(f"position[]={p}" for p in SKILL + ["K"])
    try:
        data = requests.get(f"https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&{pos}", timeout=30).json()
    except Exception:
        return {}
    out = {}
    for r in data:
        g = sp.get(str(r.get("player_id")), {}).get("gsis_id")
        pts = (r.get("stats") or {}).get("pts_ppr")
        if g and pts:
            out[g] = pts
    return out


def _save(df, season, week, cons, open_t=None):
    d = df[df.pos.isin(SKILL + ["K"])].copy()
    d["cons"] = d.player_id.map(cons)
    d = d[d.cons.notna()]
    if d.empty:
        return 0
    d["my_rank"] = d.groupby("pos").proj.rank(ascending=False, method="first")
    d["cons_rank"] = d.groupby("pos").cons.rank(ascending=False, method="first")
    if open_t is not None:  # only games that haven't kicked off: rows freeze at kickoff, no hindsight
        d = d[d.team.isin(open_t)]
    c = con()
    for r in d.itertuples():
        c.execute("INSERT OR REPLACE INTO snaps(season,week,player_id,player,pos,team,my_proj,my_rank,cons_proj,cons_rank) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (int(season), int(week), r.player_id, r.player, r.pos, r.team, float(r.proj), int(r.my_rank), float(r.cons), int(r.cons_rank)))
    c.commit()
    return len(d)


def snapshot(df, season, week, sp, open_t):
    return _save(df, season, week, consensus(season, week, sp), open_t)


def grade(ps, fin):
    """Grade each player as soon as HIS team's game is final (no waiting for the whole week)."""
    c = con()
    pts = ps.set_index(["season", "week", "player_id"]).pts
    for r in pd.read_sql("SELECT season,week,player_id,team FROM snaps WHERE actual IS NULL", c).itertuples():
        if (r.season, r.week, r.team) in fin:
            c.execute("UPDATE snaps SET actual=? WHERE season=? AND week=? AND player_id=?",
                      (float(pts.get((r.season, r.week, r.player_id), 0.0)), int(r.season), int(r.week), r.player_id))
    c.commit()


def backfill(ps, ts, sch, season, upto, sp):
    """Experimental: rebuild earlier weeks (leak-free model) vs Sleeper's stored projections."""
    c, n = con(), 0
    for wk in range(1, upto):
        if c.execute("SELECT 1 FROM snaps WHERE season=? AND week=? LIMIT 1", (season, wk)).fetchone():
            continue
        cons = consensus(season, wk, sp)
        act = ps[(ps.season == season) & (ps.week == wk)].set_index("player_id").pts
        common = [p for p in cons if p in act.index]
        # old projections that sit too close to the real scores were overwritten with results: skip
        if len(common) < 30 or np.mean([abs(cons[p] - act[p]) for p in common]) < 2.5:
            continue
        n += bool(_save(run(ps, ts, sch, season, wk), season, wk, cons))
    return n


def reset():
    c = con()
    c.execute("DELETE FROM snaps")
    c.commit()


def results(gap=8, thr=5.0):
    d = pd.read_sql("SELECT * FROM snaps WHERE actual IS NOT NULL", con())
    if d.empty:
        return None
    d["edge"] = d.cons_rank - d.my_rank  # + = this app liked him more than the other site
    d["delta"] = d.actual - d.cons_proj  # + = beat the other site's projection
    t = np.where(d.pos == "K", .6 * thr, thr)
    chalk = d.pos.map({"QB": 8, "RB": 12, "WR": 12, "TE": 6, "K": 6})  # skip obvious studs
    cols = ["season", "week", "player", "pos", "team", "my_rank", "cons_rank", "my_proj", "cons_proj", "actual"]
    return dict(
        hits=d[(d.edge >= gap) & (d.delta >= t) & (d.cons_rank > chalk)].sort_values("delta", ascending=False)[cols],
        misses=d[(d.edge >= gap) & (d.delta <= -t)].sort_values("delta")[cols],
        mae_my=(d.my_proj - d.actual).abs().mean(), mae_cons=(d.cons_proj - d.actual).abs().mean(), n=len(d))
