"""Own-model fantasy engine: context-weighted history + matchup + game environment + injuries."""
import datetime as dt
import sqlite3

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
            qb=s[f"{me}_qb_id"], coach=s[f"{me}_coach"], pf=s[f"{me}_score"], pa=s[f"{op}_score"], roof=s.roof))

    tg = pd.concat([side("home", "away", 1), side("away", "home", -1)], ignore_index=True)
    tg["t"] = tg.season * 100 + tg.week
    return tg


def next_week(sch, season):
    n = team_games(sch)
    n = n[(n.season == season) & n.pf.isna()]
    return int(n.week.min()) if len(n) else None


def completed(sch):
    g = team_games(sch).groupby(["season", "week"]).pf.apply(lambda s: s.notna().all())
    return set(g[g].index)


def sleeper_players():
    try:
        return requests.get(f"{SLEEPER}/players/nfl", timeout=40).json()
    except Exception:
        return {}


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
    try:
        teams = requests.get(f"{SLEEPER}/league/{league_id}/rosters", timeout=20).json()
    except Exception:
        return set()
    ids = {i for t in teams for i in (t.get("players") or [])}
    return {sp.get(i, {}).get("gsis_id") or i for i in ids}


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


def snapshot(df, season, week, sp):
    """Freeze this week's rankings next to the other site's. Never overwritten."""
    cons = consensus(season, week, sp)
    d = df[df.pos.isin(SKILL + ["K"])].copy()
    d["cons"] = d.player_id.map(cons)
    d = d[d.cons.notna()]
    if d.empty:
        return
    d["my_rank"] = d.groupby("pos").proj.rank(ascending=False, method="first")
    d["cons_rank"] = d.groupby("pos").cons.rank(ascending=False, method="first")
    c = con()
    for r in d.itertuples():
        c.execute("INSERT OR IGNORE INTO snaps(season,week,player_id,player,pos,team,my_proj,my_rank,cons_proj,cons_rank) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (int(season), int(week), r.player_id, r.player, r.pos, r.team, float(r.proj), int(r.my_rank), float(r.cons), int(r.cons_rank)))
    c.commit()


def grade(ps, done):
    c = con()
    pts = ps.set_index(["season", "week", "player_id"]).pts
    for r in pd.read_sql("SELECT season,week,player_id FROM snaps WHERE actual IS NULL", c).itertuples():
        if (r.season, r.week) in done:
            c.execute("UPDATE snaps SET actual=? WHERE season=? AND week=? AND player_id=?",
                      (float(pts.get((r.season, r.week, r.player_id), 0.0)), int(r.season), int(r.week), r.player_id))
    c.commit()


def results(gap=8):
    d = pd.read_sql("SELECT * FROM snaps WHERE actual IS NOT NULL", con())
    if d.empty:
        return None
    d["act_rank"] = d.groupby(["season", "week", "pos"]).actual.rank(ascending=False, method="min")
    d["edge"] = d.cons_rank - d.my_rank   # + = this app liked him more than the other site
    d["beat"] = d.cons_rank - d.act_rank  # + = finished above where the other site ranked him
    chalk = d.pos.map({"QB": 8, "RB": 12, "WR": 12, "TE": 6, "K": 6})  # skip obvious studs
    cols = ["season", "week", "player", "pos", "team", "my_rank", "cons_rank", "act_rank", "actual"]
    return dict(
        hits=d[(d.edge >= gap) & (d.beat >= 5) & (d.cons_rank > chalk)].sort_values("beat", ascending=False)[cols],
        misses=d[(d.edge >= gap) & (d.beat <= -5)].sort_values("beat")[cols],
        mae_my=(d.my_rank - d.act_rank).abs().mean(), mae_cons=(d.cons_rank - d.act_rank).abs().mean(), n=len(d))
