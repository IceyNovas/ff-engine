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
AVG, SHRINK = 22.5, 1.0
DECAY, XFP_W, MU = .8, .3, .25  # backtested: recency decay, share of ACTUAL pts in the baseline (rest = expected pts), matchup weight
START = {"QB": 16, "RB": 10.5, "WR": 10.5, "TE": 8.5, "K": 8, "DST": 7}  # rough weekly "startable" lines; edit for your league
VAC = {"RB": .45, "TE": .30, "WR": .25}  # share of an injured player's role his teammates absorb
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


def prep(ps, ex=None):
    ps = ps[(ps.season_type == "REG") & ps.position.isin(SKILL + ["K"])].copy()
    ps[KCOLS] = ps[KCOLS].fillna(0)
    kick = (3 * (ps.fg_made_0_19 + ps.fg_made_20_29 + ps.fg_made_30_39) + 4 * ps.fg_made_40_49
            + 5 * (ps.fg_made_50_59 + ps.fg_made_60_) + ps.pat_made - ps.fg_missed - ps.pat_missed)
    ps["pts"] = np.where(ps.position == "K", kick, ps.fantasy_points_ppr.fillna(0))
    ps["t"] = ps.season * 100 + ps.week
    ps = ps.merge(ex["xfp"], on=["season", "week", "player_id"], how="left") if ex and "xfp" in ex else ps.assign(xfp=np.nan)
    ps["adj"] = np.where(ps.xfp.notna() & (ps.position != "K"), XFP_W * ps.pts + (1 - XFP_W) * ps.xfp, ps.pts)  # blend actual + usage-expected
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
    """Name key that ignores punctuation and Jr/Sr/II/III so 'Kenneth Walker III' == 'Kenneth Walker'."""
    toks = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " ")).split()
    return "".join(t for t in toks if t not in ("jr", "sr", "ii", "iii", "iv", "v"))


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
        keys.add(nk(f"{p.get('first_name', '')} {p.get('last_name', '')}") + "|" + str(pos))
    return keys, len(ids), None


def league_info(lid):
    try:
        r = requests.get(f"{SLEEPER}/league/{lid.strip()}", timeout=15).json()
        return r if isinstance(r, dict) and r.get("league_id") else None
    except Exception:
        return None


def user_leagues(username, season):
    """{league name: league_id} for this season (league IDs change every year)."""
    try:
        u = requests.get(f"{SLEEPER}/user/{username.strip()}", timeout=15).json()
        L = requests.get(f"{SLEEPER}/user/{u['user_id']}/leagues/nfl/{season}", timeout=15).json()
        return {f"{l['name']}": l["league_id"] for l in L}
    except Exception:
        return {}


def load_extra(season):
    """Optional extra sources; any that fail to load are skipped and reported in the Data check."""
    yrs, ex = [season - 2, season - 1, season], {}
    try:
        o = _each(nfl.load_ff_opportunity, yrs)
        o["season"], o["week"] = o.season.astype(int), o.week.astype(int)
        ex["xfp"] = o[["season", "week", "player_id", "total_fantasy_points_exp"]].rename(columns={"total_fantasy_points_exp": "xfp"})
    except Exception:
        pass
    try:
        ro = nfl.load_rosters([season]).to_pandas()
        ex["roster"] = ro[ro.week >= ro.week.max() - 1].sort_values("week").drop_duplicates("gsis_id", keep="last")[["gsis_id", "team", "status", "sleeper_id"]]  # 2 weeks so bye-week teams are kept
    except Exception:
        pass
    try:
        ids = nfl.load_ff_playerids().to_pandas()[["gsis_id", "sleeper_id", "pfr_id"]]
        ex["ids"] = ids
        sn = nfl.load_snap_counts([season]).to_pandas()
        sn = sn[sn.offense_pct.notna()].merge(ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")[["pfr_id", "gsis_id"]],
                                              left_on="pfr_player_id", right_on="pfr_id").sort_values("week")
        sc = 100 if sn.offense_pct.max() <= 1.5 else 1
        ex["snaps"] = {g: (p[-2:].mean() * sc, (p[-2:].mean() - p[:-2][-3:].mean()) * sc if len(p) > 2 else 0.0)
                       for g, p in ((g, grp.offense_pct.values) for g, grp in sn.groupby("gsis_id"))}
    except Exception:
        pass
    return ex


def progress(sch):
    tg = team_games(sch)
    return {(int(s), int(w)): (int(g.pf.notna().sum()), len(g)) for (s, w), g in tg.groupby(["season", "week"])}


def audit(raw, ps, sch, df, ex, season, week):
    """Automatic data checks -> [(ok, message)]."""
    tg, out = team_games(sch), []
    cur = tg[(tg.season == season) & tg.pf.notna()]
    miss = set(zip(cur.season, cur.week, cur.team)) - finished(sch, ps)
    out.append((not miss, f"Stats cover every finished {season} game" if not miss else f"{len(miss)} finished team-games not in the stats feed yet (feed lags a few hours)"))
    r = raw[(raw.season == season) & raw.position.isin(SKILL)].fillna(0)
    calc = (r.passing_yards * .04 + r.passing_tds * 4 - r.passing_interceptions * 2 + r.rushing_yards * .1 + r.rushing_tds * 6 + r.receptions
            + r.receiving_yards * .1 + r.receiving_tds * 6 - (r.rushing_fumbles_lost + r.receiving_fumbles_lost + r.sack_fumbles_lost) * 2
            + (r.passing_2pt_conversions + r.rushing_2pt_conversions + r.receiving_2pt_conversions) * 2)
    pct = ((calc - r.fantasy_points_ppr).abs() < .5).mean() * 100
    out.append((pct >= 99, f"Scoring check: PPR points recomputed from raw stats match for {pct:.1f}% of {len(r)} player-games"))
    dup = int(raw.duplicated(["season", "week", "player_id"]).sum())
    out.append((dup == 0, "No duplicate player-weeks" if not dup else f"{dup} duplicate player-weeks"))
    nx = tg[(tg.season == season) & (tg.week == week)]
    bad = int(nx[["implied", "coach", "kick"]].isna().any(axis=1).sum())
    out.append((bad == 0, f"Week {week}: Vegas lines, coaches and kickoff times present for all {len(nx) // 2} games" if not bad else f"{bad} teams missing a line, coach or kickoff time"))
    mp = {p: sorted(set(nx.team) - set(df[df.pos == p].team)) for p in ("QB", "K", "DST")}
    out.append((not any(mp.values()), "Every team playing has a QB, K and D/ST projection" if not any(mp.values()) else f"Teams missing rows: {mp}"))
    rel = ps[(ps.season == season) & (ps.week >= week - 3) & (ps.week < week) & (ps.pts >= 5) & ps.position.isin(SKILL)].drop_duplicates("player_id")
    mis = rel[~rel.player_id.isin(set(df.player_id)) & rel.team.isin(set(nx.team))]
    out.append((len(mis) <= .03 * max(len(rel), 1), f"{100 * (1 - len(mis) / max(len(rel), 1)):.0f}% of players with a 5+ pt game in the last 3 weeks are in the pool"
                + (f" (not in pool, likely cut/IR: {', '.join(mis.player_display_name.head(5))})" if len(mis) else "")))
    sk = ps[(ps.season == season) & ps.position.isin(SKILL)]
    out.append((sk.xfp.notna().mean() > .8, f"Expected-points (usage) data joined for {sk.xfp.notna().mean() * 100:.0f}% of {season} player-games"))
    out.append((df.sid.notna().mean() > .95, f"{df.sid.notna().mean() * 100:.0f}% of the pool has a Sleeper ID (used to hide rostered players)"))
    out.append(("roster" in ex, "Active rosters loaded (cut/practice-squad players excluded)" if "roster" in ex else "Rosters failed to load: using recent-games rule instead"))
    return out


def tag(r):
    """Waiver label: Instant starter / Week use / Stash, with the reason."""
    s, lift = START[r.pos], r.env + r.mu + r.vac
    if r.mult == 0:
        return "", ""
    if r.pos in ("K", "DST"):
        return ("Week use", f"Streamer: projects {r.proj:.1f} (starter line ~{s:g}). K/DST value is mostly matchup, so re-check weekly.") if r.proj >= s else ("", "")
    if r.proj >= s and r.base >= .85 * s and r.games >= 3:
        return "Instant starter", f"Projects {r.proj:.1f} vs a ~{s:g} starter line on a steady role ({r.base:.1f} baseline over {r.games} games)."
    if r.proj >= .85 * s:
        if r.vac >= 1.5:
            return "Week use", f"Fill-in only: {r.vf} is out (+{r.vac:.1f}). The role may vanish when he returns."
        if lift >= 1.5:
            return "Week use", f"Matchup/game-script boost (+{lift:.1f}) puts him over the line this week only."
        return "Week use", f"Usable this week, but thin history ({r.games} games) or modest baseline ({r.base:.1f}). Don't count on it long-term."
    why = []
    if r.trend >= 2:
        why.append(f"scoring is climbing (+{r.trend:.1f} over baseline, last 3 games)")
    if r.snap_trend >= 8:
        why.append(f"snap share up {r.snap_trend:.0f} pts")
    if r.xfp4 - r.act4 >= 2:
        why.append(f"his usage says {r.xfp4:.1f} pts/game but he's scoring {r.act4:.1f} (due to bounce back)")
    if r.proj >= .5 * s and why:
        return "Stash", f"Not startable yet ({r.proj:.1f} vs ~{s:g}), but " + "; ".join(why) + "."
    return "", ""


# ---------- projections ----------
def wts(n, same_qb, same_coach):
    w = np.maximum(DECAY ** np.arange(n)[::-1], .15)  # recency, but old seasons never fully vanish
    if same_qb is not None:
        w = w * np.where(same_qb, 1, .35)  # games with a different QB count far less
    return w * np.where(same_coach, 1, .6)  # ...and so do games under a different play-caller


def run(ps, ts, sch, season, week, inj=None, ex=None):
    inj, ex = inj or {}, ex or {}
    tg = team_games(sch)
    now = season * 100 + week
    past = ps[ps.t < now].merge(tg[["season", "week", "team", "implied", "qb", "coach"]], on=["season", "week", "team"], how="left")
    nxt = tg[tg.t == now].set_index("team")
    played = tg[(tg.t < now) & tg.pf.notna()].sort_values("t")
    last = played.groupby("team").tail(1).set_index("team")
    cut = played.groupby("team").t.apply(lambda s: s.nlargest(6).min())
    nm = ps.drop_duplicates("player_id", keep="last").set_index("player_id").player_display_name
    prior = past.groupby("position").pts.mean()
    ro = ex.get("roster")
    rs = ro.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id") if ro is not None else None
    sid = {}
    for d_ in (ex.get("ids"), ro):
        if d_ is not None:
            for g_, s_ in zip(d_.gsis_id, d_.sleeper_id):
                if pd.notna(g_) and pd.notna(s_):
                    sid[g_] = str(int(float(s_)))
    snaps = ex.get("snaps", {})

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
        pos = r.position
        if rs is not None:  # roster-based pool: active/inactive/IR only; cut, practice squad, retired excluded
            if rs.status.get(pid) not in ("ACT", "INA", "RES"):
                continue
            team = rs.team.get(pid, r.team)
        else:
            team = r.team
            if r.t < cut.get(team, 0):
                continue
        if team not in nxt.index:
            continue
        if pos == "K" and r.t < last.t.get(team, 0):  # one kicker per team: the one who kicked last
            continue
        gn = nxt.loc[team]
        qb, co = cur_qb(team), coach(team)
        sq = None if pos in ("QB", "K") else (g.qb == qb).values
        sc = (g.coach == co).values
        w, x = wts(len(g), sq, sc), g.adj.values
        base = (w @ x + SHRINK * prior[pos]) / (w.sum() + SHRINK)
        sd = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
        hi = np.average(g.implied.fillna(AVG), weights=w)
        impl = gn.implied if pd.notna(gn.implied) else hi
        env = base * ENV[pos] * (impl - hi) / AVG
        rt = ratio.get((gn.opp, pos), 1.0)
        mu = base * MU * (rt - 1)
        dome = .4 if pos == "K" and gn.roof in ("dome", "closed") else 0
        st = "ir" if (rs is not None and rs.status.get(pid) == "RES") else inj.get(pid, "")
        mult = INJ.get(st, 1)
        proj = (base + env + mu + dome) * mult
        mt = sc if sq is None else (sc & sq)
        t4 = g.tail(4)
        sn = snaps.get(pid, (np.nan, 0.0))
        use = {"WR": f"{t4.target_share.mean():.0%} target share (last 4)", "TE": f"{t4.target_share.mean():.0%} target share (last 4)",
               "RB": f"{(t4.carries + t4.targets).mean():.0f} touches+targets/game (last 4)",
               "QB": f"{t4.attempts.mean():.0f} pass att/game (last 4)", "K": f"{t4.fg_att.mean():.1f} FG att/game (last 4)"}[pos]
        why = [f"Baseline {base:.1f} pts/game from {len(g)} games: {XFP_W:.0%} actual points + {1 - XFP_W:.0%} points his usage predicts, weighted to the same QB/play-caller [nflverse stats + expected points]",
               f"Game environment: team implied {impl:.1f} vs usual {hi:.1f} ({env:+.1f}) [Vegas line via nflverse]",
               f"Matchup vs {gn.opp}: allows {rt:.0%} of avg to {pos} ({mu:+.1f}; kept small because backtests showed little signal) [nflverse stats]",
               f"Usage: {use} [nflverse]"]
        if pd.notna(t4.xfp.mean()):
            why.append(f"Last 4 games: scored {t4.pts.mean():.1f}/g vs {t4.xfp.mean():.1f}/g his usage predicts [nflverse expected points]")
        if pd.notna(sn[0]):
            why.append(f"Offensive snap share {sn[0]:.0f}% ({sn[1]:+.0f} pts vs prior weeks) [nflverse snap counts]")
        if dome:
            why.append("Dome/closed roof (+0.4) [nflverse schedule]")
        if mult < 1:
            why.append(f"Injury report: {st} (x{mult}) [Sleeper / NFL report]")
        if 0 < mt.sum() < len(g):
            why.append(f"{g.pts.values[mt].mean():.1f} ppg in {mt.sum()} matching-context games vs {g.pts.values[~mt].mean():.1f} in {(~mt).sum()} others [nflverse]")
        flags = []
        if sq is not None and sq.sum() < 6:
            flags.append(f"QB change: {nm.get(qb, '?')} ({sq.sum()} of {len(g)} games together)")
        if sc.sum() < 6:
            flags.append(f"New play-caller: {co}")
        rows.append(dict(player_id=pid, player=r.player_display_name, pos=pos, team=team, opp=gn.opp, proj=proj,
                         floor=max(proj - .8 * sd, 0), ceil=proj + .8 * sd, trend=t4.pts.tail(3).mean() - base, base=base, env=env, mu=mu,
                         vac=0.0, vf="", games=len(g), act4=t4.pts.mean(), xfp4=t4.xfp.mean(), snap=sn[0], snap_trend=sn[1],
                         sid=sid.get(pid), status=st, mult=mult, why=why, flags=flags))

    # ---- D/ST ----
    d = ts[ts.season_type == "REG"].merge(tg[["season", "week", "team", "pa", "coach", "opp_implied"]], on=["season", "week", "team"], how="left")
    d["t"] = d.season * 100 + d.week
    d = d[(d.t < now) & d.pa.notna()].copy()
    tier = d.pa.map(lambda p: next(v for cap, v in PA_TIERS if p <= cap))
    d["pts"] = (d.def_sacks + 2 * d.def_interceptions + 2 * d.fumble_recovery_opp + 6 * d.def_tds + 2 * d.def_safeties
                + 2 * (d.def_fg_blocks + d.def_punt_blocks + d.def_pat_blocks) + tier)
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
        mu = base * MU * (rt - 1)
        proj = base + env + mu
        why = [f"Baseline {base:.1f} D/ST pts from {len(g)} games [nflverse team stats]",
               f"Opponent implied only {oi:.1f} vs usual {ho:.1f} faced ({env:+.1f}) [Vegas line via nflverse]",
               f"{gn.opp} offense gives up {rt:.0%} of avg D/ST points ({mu:+.1f}) [nflverse team stats]"]
        rows.append(dict(player_id=DST_ID.get(team, team), player=f"{team} D/ST", pos="DST", team=team, opp=gn.opp, proj=proj,
                         floor=max(proj - .8 * sd, 0), ceil=proj + .8 * sd, trend=g.pts.tail(3).mean() - base, base=base, env=env, mu=mu,
                         vac=0.0, vf="", games=len(g), act4=np.nan, xfp4=np.nan, snap=np.nan, snap_trend=0.0, sid=DST_ID.get(team, team),
                         status="", mult=1, why=why, flags=[] if sc.sum() >= 6 else [f"New play-caller: {co}"]))
    out = pd.DataFrame(rows)

    # ---- teammate-out boost: healthy teammates absorb part of an injured player's role ----
    for (team, pos), g in out[out.pos.isin(VAC)].groupby(["team", "pos"]):
        for o in g[g.mult < 1].itertuples():
            lost = o.base * VAC[pos] * (1 - o.mult)
            rc = g[(g.mult >= .9) & (g.player_id != o.player_id)].nlargest(2, "base")
            if rc.empty:
                continue
            for i, share in zip(rc.index, rc.base / rc.base.sum()):
                out.at[i, "vac"] += lost * share
                out.at[i, "vf"] = o.player
                out.at[i, "why"] = out.at[i, "why"] + [f"{o.player} is {o.status or 'out'}: absorbs +{lost * share:.1f} of his role [Sleeper / NFL report]"]
    out["proj"] += out.vac * out.mult
    tg_ = [tag(r) for r in out.itertuples()]
    out["tag"], out["tag_why"] = [t[0] for t in tg_], [t[1] for t in tg_]
    return out


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
    cons = consensus(season, week, sp)
    return len(cons), _save(df, season, week, cons, open_t)


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


def weeks():
    d = pd.read_sql("SELECT DISTINCT season, week FROM snaps WHERE actual IS NOT NULL ORDER BY season DESC, week DESC", con())
    return [(int(s), int(w)) for s, w in zip(d.season, d.week)]


def snap_ranks(season, week):
    return pd.read_sql("SELECT player_id, my_rank, cons_rank FROM snaps WHERE season=? AND week=?", con(), params=(int(season), int(week)))


def results(gap=8, thr=5.0, season=None, week=None):
    d = pd.read_sql("SELECT * FROM snaps WHERE actual IS NOT NULL", con())
    if week is not None:
        d = d[(d.season == season) & (d.week == week)]
    if d.empty:
        return None
    d["edge"] = d.cons_rank - d.my_rank  # + = this app liked him more than the other site
    d["delta"] = d.actual - d.cons_proj  # + = beat the other site's projection
    t = np.where(d.pos == "K", .6 * thr, thr)
    d["finish"] = d.groupby(["season", "week", "pos"]).actual.rank(ascending=False, method="min")
    chalk = d.pos.map({"QB": 8, "RB": 12, "WR": 12, "TE": 6, "K": 6})  # skip obvious studs
    cols = ["season", "week", "player", "pos", "team", "my_rank", "cons_rank", "my_proj", "cons_proj", "actual"]
    return dict(
        hits=d[(d.edge >= gap) & (d.delta >= t) & (d.cons_rank > chalk)].sort_values("delta", ascending=False)[cols],
        misses=d[(d.edge >= gap) & (d.delta <= -t)].sort_values("delta")[cols],
        all=d, mae_my=(d.my_proj - d.actual).abs().mean(), mae_cons=(d.cons_proj - d.actual).abs().mean(), n=len(d))


# ===== UI =====
import types
m = types.SimpleNamespace(**globals())

import pandas as pd
import streamlit as st


st.set_page_config(page_title="FF Engine", layout="centered")
season = m.season_now()
ICON = {"Instant starter": "🟢", "Week use": "🟡", "Stash": "🔵"}
ORDER = {"Instant starter": 0, "Week use": 1, "Stash": 2}


@st.cache_data(ttl=900, show_spinner="Loading data and building projections...")
def build(s):
    raw, ts, sch = m.load(s)
    ex = m.load_extra(s)
    ps = m.prep(raw, ex)
    sp = m.sleeper_players()
    week = m.next_week(sch, s)
    if week is None:
        return None
    inj = m.injuries(s, week, sp)
    df = m.run(ps, ts, sch, s, week, inj, ex)
    n_cons, _ = m.snapshot(df, s, week, sp, m.open_teams(sch, s, week))  # refreshes until each team's kickoff, then freezes
    m.grade(ps, m.finished(sch, ps))  # grades each player once his game is final
    info = dict(players=len(sp), injuries=len(inj), cons=n_cons, prog=m.progress(sch), stats_week=int(ps[ps.season == s].week.max()),
                checks=[(bool(ok), msg) for ok, msg in m.audit(raw, ps, sch, df, ex, s, week)])
    return df, week, info, ps[["season", "week", "player_id", "player_display_name", "position", "team", "pts"]]


def sleeper():
    return m.sleeper_players()


out = build(season)
if out is None:
    st.info("No upcoming games found in the schedule yet.")
    st.stop()
df, week, info, src = out
st.title(f"Week {week} · {season}")
st.caption('Updates every 15 min with injuries and Vegas lines. Every "Why" lists its sources in [brackets].')
if st.button("Refresh injuries & lines"):
    st.cache_data.clear()
    st.rerun()


def table(t):
    st.dataframe(t[["player", "team", "opp", "proj", "floor", "ceil", "status"]].round(1).rename(columns={
        "player": "Player", "team": "Team", "opp": "Opp", "proj": "Proj pts", "floor": "Low", "ceil": "High", "status": "Injury"}),
        hide_index=True, use_container_width=True)


def card(r):
    badge = " 🔴 OUT" if r["mult"] == 0 else (f" 🟡 {r['status']}" if r["status"] else "")
    st.markdown(f"**{r['player']}** · {r['team']} vs {r['opp']} — **{r['proj']:.1f} pts** (range {r['floor']:.0f}–{r['ceil']:.0f}){badge}")
    if r["tag"]:
        st.markdown(f"{ICON[r['tag']]} **{r['tag']}**: {r['tag_why']}")
    for f in r["flags"]:
        st.caption("⚠️ " + f)
    for w in r["why"]:
        st.markdown("- " + w)


def top_with_why(t, n=5):
    table(t)
    for _, r in t.head(n).iterrows():
        with st.expander(f"Why {r['player']}?"):
            card(r)


tabs = st.tabs(["Start/Sit", "Waivers", "Kickers", "Defenses", "Weekly Top", "Hits & Misses"])

with tabs[0]:
    st.caption("Pick 2+ players to see who to start, or browse the top 20 by position. Low–High is the realistic range of outcomes.")
    sk = df[df.pos.isin(m.SKILL)].copy()
    sk["label"] = sk.player + " (" + sk.pos + ", " + sk.team + ")"
    pick = st.multiselect("Who are you deciding between?", sk.sort_values("proj", ascending=False).label)
    if len(pick) >= 2:
        c = sk[sk.label.isin(pick)].sort_values("proj", ascending=False)
        a, b = c.iloc[0], c.iloc[1]
        st.success(f"Start **{a.player}** over {b.player}: +{a.proj - b.proj:.1f} pts projected")
        for _, r in c.iterrows():
            card(r)
            st.divider()
    else:
        pos = st.radio("Top by position", m.SKILL, horizontal=True)
        top_with_why(sk[sk.pos == pos].sort_values("proj", ascending=False).head(20))

with tabs[1]:
    st.caption("🟢 Instant starter: start him now. 🟡 Week use: good this week only. 🔵 Stash: not startable yet but rising. "
               "Enter your Sleeper username to hide players already rostered.")
    q = st.text_input("Sleeper username or league ID (press Enter)").strip()
    lid = ""
    if q:
        if q.isdigit() and len(q) > 10:
            lid = q
        else:
            lg = m.user_leagues(q, season)
            if lg:
                lid = lg[st.selectbox("Which league?", list(lg))]
            else:
                st.error(f"Couldn't find any {season} Sleeper leagues for that username.")
    pos = st.multiselect("Positions", ["QB", "RB", "WR", "TE", "K", "DST"], default=["QB", "RB", "WR", "TE"])
    show = st.multiselect("Labels", list(ICON), default=list(ICON), format_func=lambda t: f"{ICON[t]} {t}")
    taken = pd.Series(False, index=df.index)
    if lid:
        info_l = m.league_info(lid)
        if info_l:
            st.caption(f"Reading **{info_l.get('name')}** · {info_l.get('season')} season")
            if str(info_l.get("season")) != str(season):
                st.warning(f"That's a {info_l.get('season')} league. Leagues get a new ID every season, so enter your username to pick your {season} one.")
        keys, found, err = m.rostered(lid, sleeper())
        if err:
            st.error(err)
        else:
            taken = df.player_id.isin(keys) | df.sid.isin(keys) | (df.player.map(m.nk) + "|" + df.pos).isin(keys)
            st.caption(f"{found} rostered players in this league · {int(taken.sum())} hidden below")
    else:
        st.warning("No league selected, so this list includes stars who are rostered everywhere.")
    w = df[df.pos.isin(pos) & ~taken & (df.mult > 0) & df.tag.isin(show)].copy()
    w["score"] = w.proj + .5 * w.trend.clip(-4, 4)
    w["o"] = w.tag.map(ORDER)
    w = w.sort_values(["o", "score"], ascending=[True, False]).head(25)
    st.dataframe(pd.DataFrame({"Label": w.tag.map(lambda t: f"{ICON[t]} {t}"), "Player": w.player, "Pos": w.pos, "Team": w.team,
                               "Opp": w.opp, "Proj": w.proj.round(1)}), hide_index=True, use_container_width=True)
    for _, r in w.head(10).iterrows():
        with st.expander(f"Why {r['player']}?"):
            card(r)

with tabs[2]:
    st.caption("Best kickers this week. Based on how often their team scores and gets into field-goal range.")
    top_with_why(df[df.pos == "K"].sort_values("proj", ascending=False).head(12))

with tabs[3]:
    st.caption("Best defenses this week. Based on the opponent's scoring, sacks allowed and turnovers.")
    top_with_why(df[df.pos == "DST"].sort_values("proj", ascending=False).head(12))

with tabs[4]:
    st.caption("Top scorers of each week from the full NFL stat feed: every player, not just ones the app tracked. "
               "Shows where I and Sleeper ranked them beforehand, if saved.")
    wks = sorted({w_ for s_, w_ in info["prog"] if s_ == season and info["prog"][(s_, w_)][0] > 0}, reverse=True)
    if not wks:
        st.info("No finished games yet this season.")
    else:
        wk = st.selectbox("Week", wks, format_func=lambda w_: f"Week {w_}", key="topweek")
        fin, tot = info["prog"][(season, wk)]
        st.caption(f"{fin} of {tot} teams have played" + (" (final)" if fin == tot else " so far"))
        tp = st.radio("Position", ["QB", "RB", "WR", "TE", "K"], horizontal=True, index=2, key="toppos")
        t = src[(src.season == season) & (src.week == wk) & (src.position == tp)].nlargest(15, "pts").merge(
            m.snap_ranks(season, wk), on="player_id", how="left")
        t.insert(0, "Rank", range(1, len(t) + 1))
        st.dataframe(t[["Rank", "player_display_name", "team", "pts", "my_rank", "cons_rank"]].rename(columns={
            "player_display_name": "Player", "team": "Team", "pts": "Scored", "my_rank": "My rank", "cons_rank": "Their rank"}).round(1),
            hide_index=True, use_container_width=True)

with tabs[5]:
    st.caption("Before each game starts, the app saves its ranking next to Sleeper's. After the game, it grades both against what the player actually scored.")
    wks = m.weeks()
    if wks:
        lab = {(f"Week {w}" if s == season else f"{s} Week {w}"): (s, w) for s, w in wks}
        pick = st.selectbox("Week", list(lab) + ["All weeks"])
        with st.expander("Adjust strictness"):
            gap = st.slider("I rank him at least this many spots above the other site", 2, 20, 8)
            thr = st.slider("He must beat (or miss) their projection by this many points", 2, 10, 5)
        res = m.results(gap, thr) if pick == "All weeks" else m.results(gap, thr, *lab[pick])
        a1, a2 = st.columns(2)
        a1.metric("My projections missed by", f"{res['mae_my']:.1f} pts", help="Average gap between my projection and the actual score. Lower is better.")
        a2.metric("Other site missed by", f"{res['mae_cons']:.1f} pts")
        diff = res["mae_cons"] - res["mae_my"]
        st.caption(f"{res['n']} players graded. " + (f"My model is closer by {diff:.2f} pts per player." if diff > 0 else f"The other site is closer by {-diff:.2f} pts per player."))
        names = {"player": "Player", "pos": "Pos", "team": "Team", "week": "Wk", "my_rank": "My rank", "cons_rank": "Their rank",
                 "my_proj": "My proj", "cons_proj": "Their proj", "actual": "Scored"}

        def lst(title, note, d):
            st.subheader(title)
            st.caption(note)
            if d.empty:
                st.write("None with these settings. Try loosening the sliders above.")
            else:
                d = d.drop(columns=["season"] + ([] if pick == "All weeks" else ["week"]))
                st.dataframe(d.rename(columns=names).round(1), hide_index=True, use_container_width=True)

        lst("✅ Hits", "Players I ranked well above the other site (not obvious stars) who beat their projection.", res["hits"])
        lst("❌ Misses", "Players I ranked well above the other site who fell short of their projection.", res["misses"])
    else:
        st.info("Nothing graded yet. Players show up here as their games finish, or use Backfill below.")
    with st.expander("History tools"):
        if st.button("Backfill earlier weeks (experimental)"):
            with st.spinner("Rebuilding past weeks..."):
                ps_raw, ts, sch = m.load(season)
                ps = m.prep(ps_raw)
                n = m.backfill(ps, ts, sch, season, week, sleeper())
                m.grade(ps, m.finished(sch, ps))
            st.success(f"Added {n} week(s). Reload the tab.")
        if st.button("Reset all history"):
            m.reset()
            st.rerun()


with st.expander("Data check (automatic)"):
    for ok, msg in info["checks"]:
        st.write(("✅ " if ok else "⚠️ ") + msg)
    st.write(f"{'✅' if info['players'] else '⚠️'} Sleeper player list: {info['players']:,} players · "
             f"{'✅' if info['injuries'] else '⚠️'} live injury flags: {info['injuries']} · "
             f"{'✅' if info['cons'] else '⚠️'} Sleeper projections this week: {info['cons']} players · stats through Week {info['stats_week']}")

with st.expander("Sources & method"):
    st.markdown("""
**Data sources**
- **nflverse** (github.com/nflverse): game stats, team stats, schedules with Vegas lines, active rosters, snap counts, and expected fantasy points (how many points a player's usage should produce).
- **Sleeper API**: live injury status, your league's rosters, and player IDs. Sleeper's weekly projections are used only as the "other site" in Hits & Misses.

**How a projection is built**
1. **Baseline:** recent games, newest weighted most, and games with the same QB and play-caller count more. It blends 30% actual points with 70% expected points from usage, so touchdown luck matters less.
2. **Adjustments:** game environment (Vegas implied team total), a small matchup adjustment, and a share of any injured teammate's role.
3. **Injury multiplier:** Out 0, Doubtful 0.25, Questionable 0.92.

**How accurate is it?** Tested against 5 past weeks (2025 weeks 10, 12, 14 and 2026 weeks 5, 6): the average miss was 4.54 pts per player versus 4.58 for a plain "last 4 games" average, and rank correlation was 0.67 versus 0.64. That's a small edge. Matchup adjustments added almost nothing, so they stay small. Single-week fantasy scoring is very noisy, so treat projections as an edge, not a certainty.

**Waiver labels** (starter line is about QB 16, RB/WR 10.5, TE 8.5, K 8, D/ST 7 pts)
- 🟢 **Instant starter:** projects above the line on a steady role (3+ games).
- 🟡 **Week use:** above the line only because of a matchup, game script or injured teammate, or on a thin role. K/DST are streamers.
- 🔵 **Stash:** below the line but rising: scoring up, snap share up, or usage far above his points.
""")
