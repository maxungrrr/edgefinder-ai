
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="EdgeFinder AI", page_icon="📈", layout="wide")

# ----------------------------
# Configuration
# ----------------------------
SPORTS = {
    "NBA": "basketball_nba",
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
}

PROP_MARKETS = {
    "NBA": ["player_points", "player_rebounds", "player_assists"],
    "NFL": ["player_pass_yds", "player_rush_yds", "player_rec_yds"],
    "MLB": ["batter_hits", "batter_total_bases", "pitcher_strikeouts"],
}

API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


# ----------------------------
# Utilities
# ----------------------------
def american_to_implied(odds: float) -> float:
    if odds >= 100:
        return 100 / (odds + 100)
    return (-odds) / ((-odds) + 100)


def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1 + odds / 100
    return 1 + 100 / (-odds)


def expected_value(probability: float, odds: float) -> float:
    """Expected profit per $1 risked."""
    dec = american_to_decimal(odds)
    return probability * (dec - 1) - (1 - probability)


def pretty_market(market: str) -> str:
    return market.replace("player_", "").replace("_", " ").title()


def demo_data():
    # Deliberately labeled DEMO data so the app never presents fabricated
    # live odds as real betting information.
    rows = [
        ["Demo Player A", "Boston", "New York", "player_points", 24.5, -110, 0.587],
        ["Demo Player B", "Dallas", "Phoenix", "player_rebounds", 7.5, -115, 0.571],
        ["Demo Player C", "Los Angeles", "Denver", "player_assists", 5.5, +105, 0.532],
        ["Demo Player D", "Milwaukee", "Miami", "player_points", 21.5, -105, 0.565],
        ["Demo Player E", "Minnesota", "Oklahoma City", "player_rebounds", 9.5, +100, 0.541],
        ["Demo Player F", "New York", "Chicago", "player_assists", 6.5, -120, 0.603],
    ]
    return pd.DataFrame(
        rows,
        columns=["player", "team", "opponent", "market", "line", "odds", "model_prob"],
    )


def fetch_odds(api_key: str, sport: str, markets: list[str]) -> pd.DataFrame:
    """
    Fetch current odds. The Odds API may return player-prop markets depending
    on sport, bookmaker coverage, and your plan.
    """
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm,bet365",
    }
    response = requests.get(API_URL.format(sport=sport), params=params, timeout=20)
    response.raise_for_status()
    events = response.json()

    rows = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("title", bookmaker.get("key", "Unknown"))
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    # Player props generally have description = player name.
                    player = outcome.get("description") or outcome.get("name", "")
                    side = outcome.get("name", "")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if point is None or price is None:
                        continue
                    rows.append(
                        {
                            "player": player,
                            "team": "",
                            "opponent": "",
                            "market": market_key,
                            "line": float(point),
                            "odds": float(price),
                            "side": side,
                            "book": book,
                            "home": home,
                            "away": away,
                        }
                    )

    return pd.DataFrame(rows)


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["implied_prob"] = out["odds"].apply(american_to_implied)

    # MVP model:
    # For demo records, model_prob is supplied.
    # For live records, this version intentionally does not fabricate a
    # probability from insufficient data. Live modeling is the next phase.
    if "model_prob" not in out:
        out["model_prob"] = np.nan

    out["edge"] = out["model_prob"] - out["implied_prob"]
    out["ev"] = [
        expected_value(p, o) if pd.notna(p) else np.nan
        for p, o in zip(out["model_prob"], out["odds"])
    ]
    out["edge_pct"] = out["edge"] * 100
    out["ev_pct"] = out["ev"] * 100
    out["rating"] = pd.cut(
        out["edge_pct"],
        bins=[-np.inf, 0, 2, 4, 7, np.inf],
        labels=["PASS", "WATCH", "LEAN", "BET", "STRONG"],
    )
    return out.sort_values(["edge_pct", "ev_pct"], ascending=False)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ EdgeFinder AI")
sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()))
mode = st.sidebar.radio("Data mode", ["Demo mode", "Live odds"])

st.sidebar.divider()
st.sidebar.caption("MVP: odds → probability → edge → EV → ranking")

api_key = ""
if mode == "Live odds":
    api_key = st.sidebar.text_input(
        "The Odds API key",
        value="",
        type="password",
        help="You can later store this in Streamlit Secrets instead of entering it here.",
    )

# ----------------------------
# Header
# ----------------------------
st.title("📈 EdgeFinder AI")
st.subheader(f"{sport_name} Player Props")
st.write(
    "A research and paper-trading dashboard that compares sportsbook prices "
    "with a model probability and ranks potential edges."
)

if mode == "Demo mode":
    st.info(
        "DEMO MODE: the displayed players, lines, and probabilities are synthetic "
        "examples. They are not live sportsbook odds."
    )

# ----------------------------
# Data
# ----------------------------
if mode == "Demo mode":
    df = demo_data()
    df["book"] = "Demo Book"
    df["side"] = np.where(df["model_prob"] >= 0.55, "Over", "Under")
else:
    if not api_key:
        st.warning("Enter an Odds API key in the sidebar to load live odds.")
        st.stop()

    try:
        df = fetch_odds(api_key, SPORTS[sport_name], PROP_MARKETS[sport_name])
        if df.empty:
            st.warning(
                "No player-prop markets were returned. This can happen because of "
                "sport timing, bookmaker coverage, or API-plan market availability."
            )
            st.stop()

        # We do NOT pretend live odds alone are a predictive model.
        df["model_prob"] = np.nan
        st.warning(
            "Live odds are loaded, but the predictive model is not yet trained. "
            "The current release will therefore not manufacture an edge."
        )
    except Exception as exc:
        st.error(f"Could not load live odds: {exc}")
        st.stop()

scored = score_candidates(df)

# ----------------------------
# KPI cards
# ----------------------------
if scored["model_prob"].notna().any():
    best = scored.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best edge", f"{best['edge_pct']:.1f}%")
    c2.metric("Model probability", f"{best['model_prob']:.1%}")
    c3.metric("Implied probability", f"{best['implied_prob']:.1%}")
    c4.metric("Expected value", f"{best['ev_pct']:.1f}%")
else:
    st.info("Model probabilities will appear after the modeling phase is connected.")

# ----------------------------
# Filters
# ----------------------------
st.subheader("🔎 Candidate Bets")
min_edge = st.slider("Minimum model edge", 0.0, 15.0, 3.0, 0.5)
books = sorted(scored["book"].dropna().unique().tolist())
selected_books = st.multiselect("Sportsbooks", books, default=books)

filtered = scored[scored["book"].isin(selected_books)].copy()
if filtered["model_prob"].notna().any():
    filtered = filtered[filtered["edge_pct"] >= min_edge]

display_cols = [
    "player", "market", "side", "line", "odds", "book",
    "model_prob", "implied_prob", "edge_pct", "ev_pct", "rating"
]
display_cols = [c for c in display_cols if c in filtered.columns]

styled = filtered[display_cols].copy()
if "model_prob" in styled:
    styled["model_prob"] = styled["model_prob"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
if "implied_prob" in styled:
    styled["implied_prob"] = styled["implied_prob"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
if "edge_pct" in styled:
    styled["edge_pct"] = styled["edge_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
if "ev_pct" in styled:
    styled["ev_pct"] = styled["ev_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

st.dataframe(styled, use_container_width=True, hide_index=True)

# ----------------------------
# Methodology
# ----------------------------
with st.expander("How EdgeFinder calculates value"):
    st.markdown(
        """
**Implied probability**

- Positive odds: `100 / (odds + 100)`
- Negative odds: `-odds / (-odds + 100)`

**Expected value**

`EV = model probability × profit if successful − probability of loss`

The most important future upgrade is the predictive model. It will estimate
player performance from historical data rather than using recent averages alone.

**Rule:** the app should never label a wager as +EV unless the model probability
is actually available and exceeds the sportsbook's implied probability.
"""
    )

st.caption(
    f"Last app refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)
