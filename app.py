"""Run:  pip install -r requirements.txt  &&  streamlit run app.py"""
import streamlit as st

import model as m

st.set_page_config(page_title="FF Engine", layout="centered")
season = m.season_now()


@st.cache_data(ttl=900, show_spinner="Crunching projections...")
def build(s):
    ps_raw, ts, sch = m.load(s)
    sp, ps = m.sleeper_players(), m.prep(ps_raw)
    week = m.next_week(sch, s)
    if week is None:
        return None
    df = m.run(ps, ts, sch, s, week, m.injuries(s, week, sp))
    m.snapshot(df, s, week, sp)  # freezes my rankings vs the other site, once per player-week
    m.grade(ps, m.completed(sch))  # grades finished weeks
    return df, week


@st.cache_data(ttl=1800)
def sleeper():
    return m.sleeper_players()


out = build(season)
if out is None:
    st.info("No upcoming games found in the schedule yet.")
    st.stop()
df, week = out
st.title(f"Week {week} · {season}")
if st.button("Refresh injuries & lines"):
    st.cache_data.clear()
    st.rerun()


def table(t):
    st.dataframe(t[["player", "team", "opp", "proj", "floor", "ceil", "status"]].round(1),
                 hide_index=True, use_container_width=True)


def card(r):
    tag = " 🔴 OUT" if r["mult"] == 0 else (f" 🟡 {r['status']}" if r["status"] else "")
    st.markdown(f"**{r['player']}** · {r['team']} vs {r['opp']} — **{r['proj']:.1f} pts** "
                f"(range {r['floor']:.0f}–{r['ceil']:.0f}){tag}")
    for f in r["flags"]:
        st.caption("⚠️ " + f)
    for w in r["why"]:
        st.markdown("- " + w)


def top_with_why(t, n=5):
    table(t)
    for _, r in t.head(n).iterrows():
        with st.expander(f"Why {r['player']}?"):
            card(r)


tabs = st.tabs(["Start/Sit", "Kickers", "Defenses", "Waivers", "Hits & Misses"])

with tabs[0]:
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
    top_with_why(df[df.pos == "K"].sort_values("proj", ascending=False).head(12))

with tabs[2]:
    top_with_why(df[df.pos == "DST"].sort_values("proj", ascending=False).head(12))

with tabs[3]:
    lid = st.text_input("Sleeper league ID (optional, hides rostered players)")
    taken = m.rostered(lid, sleeper()) if lid else set()
    pos = st.multiselect("Positions", ["QB", "RB", "WR", "TE", "K", "DST"], default=["RB", "WR", "TE"])
    w = df[df.pos.isin(pos) & ~df.player_id.isin(taken) & (df.mult > 0)].copy()
    w["score"] = w.proj + .5 * w.trend.clip(-4, 4)  # projection + breakout trend
    w = w.sort_values("score", ascending=False).head(20)
    st.caption("Waiver score = projection + half of the recent-form jump over baseline")
    top_with_why(w, 8)

with tabs[4]:
    gap = st.slider("Min rank gap vs the other site", 4, 20, 8)
    res = m.results(gap)
    if res is None:
        st.info("Your rankings get frozen next to the other site's before each week. "
                "Check back once this week's games finish and this fills in.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("My avg rank error", f"{res['mae_my']:.1f}")
        c2.metric("Other site's", f"{res['mae_cons']:.1f}", help="Lower is better; n=%d player-weeks" % res["n"])
        st.subheader("Best hits (sleepers I liked, they delivered)")
        st.dataframe(res["hits"], hide_index=True, use_container_width=True)
        st.subheader("Misses (I was higher, they flopped)")
        st.dataframe(res["misses"], hide_index=True, use_container_width=True)
