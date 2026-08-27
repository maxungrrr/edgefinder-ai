import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="EdgeFinder AI — WNBA", page_icon="🏀", layout="wide")

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_wnba"
MARKETS = {
    "Points":"player_points","1Q Points":"player_points_q1",
    "Rebounds":"player_rebounds","1Q Rebounds":"player_rebounds_q1",
    "Assists":"player_assists","1Q Assists":"player_assists_q1",
    "Threes":"player_threes","PRA":"player_points_rebounds_assists",
    "PR":"player_points_rebounds","PA":"player_points_assists","RA":"player_rebounds_assists"
}
BOOKS = ["draftkings","fanduel","betmgm","bet365","caesars","betrivers"]

def implied(odds):
    odds=float(odds)
    return 100/(odds+100) if odds>0 else (-odds)/(-odds+100)

def fair_odds(p):
    if not 0 < p < 1: return np.nan
    return -100*p/(1-p) if p >= .5 else 100*(1-p)/p

def api(url,key,params=None):
    p=dict(params or {}); p["apiKey"]=key
    r=requests.get(url,params=p,timeout=25)
    if not r.ok:
        try: msg=r.json().get("message",r.text)
        except Exception: msg=r.text
        raise RuntimeError(f"Odds API {r.status_code}: {msg}")
    return r.json()

@st.cache_data(ttl=60)
def events(key):
    return api(f"{BASE}/sports/{SPORT}/events",key)

@st.cache_data(ttl=60)
def event_odds(key,event_id,markets):
    return api(f"{BASE}/sports/{SPORT}/events/{event_id}/odds",key,{
        "regions":"us","markets":",".join(markets),"oddsFormat":"american"
    })

def flatten(e, wanted):
    rows=[]
    for b in e.get("bookmakers",[]):
        if b.get("key") not in BOOKS: continue
        for m in b.get("markets",[]):
            if m.get("key") not in wanted: continue
            for o in m.get("outcomes",[]):
                if o.get("point") is None or o.get("price") is None: continue
                rows.append({
                    "game":f"{e.get('away_team')} @ {e.get('home_team')}",
                    "player":o.get("description",""),
                    "market":m.get("key"),
                    "market_name":next((x for x,y in MARKETS.items() if y==m.get("key")),m.get("key")),
                    "side":o.get("name",""),
                    "line":float(o["point"]),
                    "odds":float(o["price"]),
                    "bookmaker":b.get("title",b.get("key"))
                })
    return pd.DataFrame(rows)

st.title("🏀 EdgeFinder AI")
st.caption("WNBA live-market scanner — Version 2")
st.info("This version loads current WNBA player-prop markets and calculates market-implied and no-vig probabilities. It does not invent an AI prediction yet.")

with st.sidebar:
    st.header("Settings")
    mode=st.radio("Data mode",["Live WNBA odds","Demo mode"])
    key=st.text_input("The Odds API key",type="password") if mode=="Live WNBA odds" else ""
    labels=st.multiselect("Markets",list(MARKETS),default=["Points","1Q Points","Rebounds","Assists","Threes"])
    wanted=[MARKETS[x] for x in labels]
    book_labels=st.multiselect("Sportsbooks",["DraftKings","FanDuel","BetMGM","bet365","Caesars","BetRivers"],
                               default=["DraftKings","FanDuel","BetMGM","bet365"])
    selected_books=set(book_labels)

if mode=="Demo mode":
    rows=[
        ["Demo Player A","Points","Over",18.5,-110,"Demo Book"],["Demo Player A","Points","Under",18.5,-110,"Demo Book"],
        ["Demo Player B","1Q Points","Over",3.5,-115,"Demo Book"],["Demo Player B","1Q Points","Under",3.5,-105,"Demo Book"],
        ["Demo Player C","Rebounds","Over",7.5,100,"Demo Book"],["Demo Player C","Rebounds","Under",7.5,-120,"Demo Book"],
    ]
    df=pd.DataFrame(rows,columns=["player","market_name","side","line","odds","bookmaker"])
    games=1
    st.warning("DEMO MODE: players, lines and odds are synthetic.")
else:
    if not key:
        st.warning("Enter your free The Odds API key in the sidebar.")
        st.stop()
    try:
        ev=events(key)
    except Exception as e:
        st.error(str(e)); st.stop()
    if not ev:
        st.warning("No current/upcoming WNBA games were returned."); st.stop()
    options={f"{e['away_team']} @ {e['home_team']} — {pd.to_datetime(e['commence_time']).strftime('%b %d %I:%M %p')}":e["id"] for e in ev}
    chosen=st.selectbox("WNBA game",list(options))
    try:
        e=event_odds(key,options[chosen],wanted)
    except Exception as ex:
        st.error(str(ex)); st.stop()
    df=flatten(e,wanted)
    games=len(ev)
    if df.empty:
        st.warning("No selected player props were returned for this game. Try another game or market."); st.stop()
    if selected_books: df=df[df["bookmaker"].isin(selected_books)].copy()

df["implied_prob"]=df["odds"].apply(implied)

c1,c2,c3,c4=st.columns(4)
c1.metric("WNBA games",games); c2.metric("Prop rows",len(df)); c3.metric("Players",df["player"].nunique()); c4.metric("Books",df["bookmaker"].nunique())

st.subheader("📊 Market Consensus")
cons=[]
for (player,market,line),g in df.groupby(["player","market_name","line"]):
    over=g[g.side.str.lower()=="over"]; under=g[g.side.str.lower()=="under"]
    for book in set(over.bookmaker)&set(under.bookmaker):
        o=over[over.bookmaker==book].iloc[0]; u=under[under.bookmaker==book].iloc[0]
        po,pu=implied(o.odds),implied(u.odds); total=po+pu
        if total:
            cons.append([player,market,line,book,po/total,pu/total])
    # handled below
cd=pd.DataFrame(cons,columns=["player","market","line","book","over_p","under_p"])
if cd.empty:
    st.warning("Not enough two-sided prices to calculate no-vig consensus.")
else:
    out=[]
    for (player,market,line),g in cd.groupby(["player","market","line"]):
        po=g.over_p.mean(); pu=g.under_p.mean()
        out += [[player,market,"Over",line,po,fair_odds(po)], [player,market,"Under",line,pu,fair_odds(pu)]]
    consensus=pd.DataFrame(out,columns=["player","market","side","line","consensus_prob","fair_odds"])
    consensus["consensus_prob"]=consensus.consensus_prob.map(lambda x:f"{x:.1%}")
    consensus["fair_odds"]=consensus.fair_odds.map(lambda x:"—" if pd.isna(x) else f"{x:+.0f}")
    st.dataframe(consensus,use_container_width=True,hide_index=True)

st.subheader("🏪 Best Available Prices")
show=df.copy()
show["implied_prob"]=show.implied_prob.map(lambda x:f"{x:.1%}")
show["odds"]=show.odds.map(lambda x:f"{x:+.0f}")
st.dataframe(show[["player","market_name","side","line","bookmaker","odds","implied_prob"]].sort_values(["player","market_name","line","side"]),
             use_container_width=True,hide_index=True)

with st.expander("What comes next"):
    st.markdown("""**Current market:** sportsbook prices, implied probabilities, no-vig consensus and fair odds.

**Next model:** player game logs, minutes, usage, recent 5/10/20-game form, opponent matchup, pace, home/away, rest, teammate availability, game environment, and quarter-specific features.

The model will produce an independent probability and compare it with the market. We will paper-track every prediction before treating it as evidence of an edge.""")

st.caption(f"Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
