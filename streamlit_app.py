import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="EdgeFinder AI — WNBA",
    page_icon="🏀",
    layout="wide"
)

# ============================================================
# CONFIG
# ============================================================

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_wnba"

MARKETS = {
    "Points": "player_points",
    "1Q Points": "player_points_q1",
    "Rebounds": "player_rebounds",
    "1Q Rebounds": "player_rebounds_q1",
    "Assists": "player_assists",
    "1Q Assists": "player_assists_q1",
    "Threes": "player_threes",
    "PRA": "player_points_rebounds_assists",
    "PR": "player_points_rebounds",
    "PA": "player_points_assists",
    "RA": "player_rebounds_assists",
}

BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "bet365",
    "caesars",
    "betrivers",
]

# ESPN public endpoints are used for historical WNBA game data.
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/wnba/scoreboard"
)

ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/wnba/summary"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def implied(odds):
    odds = float(odds)

    if odds > 0:
        return 100 / (odds + 100)

    return (-odds) / (-odds + 100)


def fair_odds(p):
    if not 0 < p < 1:
        return np.nan

    if p >= 0.5:
        return -100 * p / (1 - p)

    return 100 * (1 - p) / p


def american_to_decimal(odds):
    odds = float(odds)

    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def expected_value(probability, odds):
    decimal = american_to_decimal(odds)

    return probability * decimal - 1


def api(url, key, params=None):
    p = dict(params or {})
    p["apiKey"] = key

    r = requests.get(
        url,
        params=p,
        timeout=25
    )

    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text

        raise RuntimeError(
            f"Odds API {r.status_code}: {msg}"
        )

    return r.json()


# ============================================================
# ODDS API
# ============================================================

@st.cache_data(ttl=60)
def events(key):
    return api(
        f"{BASE}/sports/{SPORT}/events",
        key
    )


@st.cache_data(ttl=60)
def event_odds(key, event_id, markets):
    return api(
        f"{BASE}/sports/{SPORT}/events/{event_id}/odds",
        key,
        {
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
    )


def flatten(e, wanted):
    rows = []

    for b in e.get("bookmakers", []):

        if b.get("key") not in BOOKS:
            continue

        for m in b.get("markets", []):

            if m.get("key") not in wanted:
                continue

            for o in m.get("outcomes", []):

                if (
                    o.get("point") is None
                    or o.get("price") is None
                ):
                    continue

                rows.append({
                    "game": (
                        f"{e.get('away_team')} @ "
                        f"{e.get('home_team')}"
                    ),
                    "player": o.get("description", ""),
                    "market": m.get("key"),
                    "market_name": next(
                        (
                            x for x, y in MARKETS.items()
                            if y == m.get("key")
                        ),
                        m.get("key")
                    ),
                    "side": o.get("name", ""),
                    "line": float(o["point"]),
                    "odds": float(o["price"]),
                    "bookmaker": b.get(
                        "title",
                        b.get("key")
                    )
                })

    return pd.DataFrame(rows)


# ============================================================
# ESPN HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=60 * 60)
def get_wnba_games(days_back=45):

    today = datetime.now(timezone.utc).date()

    start = today - timedelta(days=days_back)

    dates = []

    current = start

    while current <= today:
        dates.append(
            current.strftime("%Y%m%d")
        )
        current += timedelta(days=1)

    all_events = []

    for date in dates:

        try:
            r = requests.get(
                ESPN_SCOREBOARD,
                params={
                    "dates": date,
                    "limit": 100
                },
                timeout=15
            )

            if not r.ok:
                continue

            data = r.json()

            for event in data.get(
                "events",
                []
            ):
                all_events.append(event)

        except Exception:
            continue

    # Remove duplicates
    unique = {}

    for event in all_events:
        unique[event.get("id")] = event

    return list(unique.values())


@st.cache_data(ttl=60 * 60)
def get_game_boxscore(event_id):

    try:

        r = requests.get(
            ESPN_SUMMARY,
            params={
                "event": event_id
            },
            timeout=20
        )

        if not r.ok:
            return None

        return r.json()

    except Exception:
        return None


def extract_player_stats(summary):

    rows = []

    if not summary:
        return rows

    boxscore = summary.get(
        "boxscore",
        {}
    )

    players = boxscore.get(
        "players",
        []
    )

    for team in players:

        team_name = team.get(
            "team",
            {}
        ).get(
            "displayName",
            ""
        )

        statistics = team.get(
            "statistics",
            []
        )

        for stat_group in statistics:

            athletes = stat_group.get(
                "athletes",
                []
            )

            labels = stat_group.get(
                "labels",
                []
            )

            for athlete in athletes:

                name = athlete.get(
                    "athlete",
                    {}
                ).get(
                    "displayName",
                    ""
                )

                stats = athlete.get(
                    "stats",
                    []
                )

                if not name or not stats:
                    continue

                values = dict(
                    zip(labels, stats)
                )

                def number(key):
                    value = values.get(key, 0)

                    try:
                        return float(
                            str(value).replace(
                                "--",
                                "0"
                            )
                        )
                    except Exception:
                        return 0.0

                rows.append({

                    "player": name,

                    "team": team_name,

                    "minutes": number("MIN"),

                    "points": number("PTS"),

                    "rebounds": number("REB"),

                    "assists": number("AST"),

                    "threes": number("3PM"),

                    "steals": number("STL"),

                    "blocks": number("BLK"),

                })

    return rows


@st.cache_data(ttl=60 * 60)
def build_historical_dataset(days_back=45):

    games = get_wnba_games(days_back)

    all_rows = []

    # Completed games only.
    for game in games:

        status = (
            game.get("status", {})
            .get("type", {})
            .get("completed", False)
        )

        if not status:
            continue

        event_id = game.get("id")

        summary = get_game_boxscore(
            event_id
        )

        stats = extract_player_stats(
            summary
        )

        for row in stats:

            row["event_id"] = event_id

            row["date"] = game.get(
                "date"
            )

            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values(
        ["player", "date"]
    )

    return df


# ============================================================
# MODEL
# ============================================================

def market_to_stat(market):

    mapping = {

        "Points": "points",
        "1Q Points": "points",

        "Rebounds": "rebounds",
        "1Q Rebounds": "rebounds",

        "Assists": "assists",

        "Threes": "threes",

        "PRA": "pra",
        "PR": "pr",
        "PA": "pa",
        "RA": "ra",
    }

    return mapping.get(
        market,
        market.lower()
    )


def add_combo_stats(df):

    df = df.copy()

    df["pra"] = (
        df["points"]
        + df["rebounds"]
        + df["assists"]
    )

    df["pr"] = (
        df["points"]
        + df["rebounds"]
    )

    df["pa"] = (
        df["points"]
        + df["assists"]
    )

    df["ra"] = (
        df["rebounds"]
        + df["assists"]
    )

    return df


def player_projection(
    historical,
    player,
    market,
    line
):

    if historical.empty:
        return None

    historical = add_combo_stats(
        historical
    )

    stat = market_to_stat(
        market
    )

    player_df = historical[
        historical["player"].str.lower()
        == player.lower()
    ].copy()

    if player_df.empty:
        return None

    player_df = player_df.sort_values(
        "date"
    )

    # Only use games before the current prediction.
    recent = player_df.tail(20)

    if recent.empty:
        return None

    values = pd.to_numeric(
        recent[stat],
        errors="coerce"
    ).dropna()

    if len(values) < 3:
        return None

    # --------------------------------------------------------
    # Weighted recent-form model
    # --------------------------------------------------------

    last5 = values.tail(5)
    last10 = values.tail(10)
    last20 = values.tail(20)

    mean5 = last5.mean()
    mean10 = last10.mean()
    mean20 = last20.mean()

    # More weight to recent performance.
    projection = (
        mean5 * 0.50
        + mean10 * 0.30
        + mean20 * 0.20
    )

    # --------------------------------------------------------
    # Minutes adjustment
    # --------------------------------------------------------

    if "minutes" in recent:

        minutes = pd.to_numeric(
            recent["minutes"],
            errors="coerce"
        ).dropna()

        if len(minutes) >= 3:

            recent_minutes = minutes.tail(5).mean()
            season_minutes = minutes.mean()

            if (
                season_minutes > 0
                and recent_minutes > 0
            ):

                ratio = (
                    recent_minutes
                    / season_minutes
                )

                # Small adjustment only.
                ratio = np.clip(
                    ratio,
                    0.85,
                    1.15
                )

                projection *= ratio

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    std = values.std()

    if pd.isna(std) or std <= 0:
        std = max(
            1.0,
            abs(projection) * 0.15
        )

    # Don't allow absurdly low variance.
    std = max(
        std,
        0.75
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    # Continuity correction.
    if line % 1 == 0.5:
        threshold = line
    else:
        threshold = line

    z = (
        threshold - projection
    ) / std

    # Normal distribution.
    over_probability = (
        0.5
        * (
            1
            - math.erf(
                z / math.sqrt(2)
            )
        )
    )

    under_probability = (
        1
        - over_probability
    )

    return {
        "projection": projection,
        "std": std,
        "over_probability":
            over_probability,
        "under_probability":
            under_probability,
        "sample_size":
            len(values),
        "last5":
            mean5,
        "last10":
            mean10,
        "last20":
            mean20,
        "minutes":
            recent["minutes"].tail(5).mean()
            if "minutes" in recent
            else np.nan,
    }


# ============================================================
# BET SCORING
# ============================================================

def score_bet(row):

    model_prob = row["model_probability"]

    market_prob = implied(
        row["odds"]
    )

    edge = (
        model_prob
        - market_prob
    )

    ev = expected_value(
        model_prob,
        row["odds"]
    )

    if edge >= 0.10:
        rating = "🔥 ELITE"

    elif edge >= 0.07:
        rating = "🔥 STRONG"

    elif edge >= 0.04:
        rating = "🟢 GOOD"

    elif edge >= 0.02:
        rating = "🟡 LEAN"

    elif edge <= -0.04:
        rating = "🔴 FADE"

    else:
        rating = "⚪ PASS"

    return pd.Series({
        "market_probability":
            market_prob,

        "edge":
            edge,

        "expected_value":
            ev,

        "rating":
            rating
    })


# ============================================================
# UI
# ============================================================

st.title(
    "🏀 EdgeFinder AI"
)

st.caption(
    "WNBA live-market scanner + statistical prediction model"
)

st.info(
    "Live mode uses real sportsbook odds and "
    "real historical WNBA player performance. "
    "The model produces an independent projection; "
    "it does not use the sportsbook's implied probability "
    "as its prediction."
)

with st.sidebar:

    st.header("Settings")

    mode = st.radio(
        "Data mode",
        ["Live WNBA odds"]
    )

    key = st.text_input(
        "The Odds API key",
        type="password"
    )

    labels = st.multiselect(
        "Markets",
        list(MARKETS),
        default=[
            "Points",
            "Rebounds",
            "Assists",
            "Threes",
            "PRA"
        ]
    )

    wanted = [
        MARKETS[x]
        for x in labels
    ]

    book_labels = st.multiselect(
        "Sportsbooks",
        [
            "DraftKings",
            "FanDuel",
            "BetMGM",
            "bet365",
            "Caesars",
            "BetRivers"
        ],
        default=[
            "DraftKings",
            "FanDuel",
            "BetMGM",
            "bet365"
        ]
    )

    selected_books = set(
        book_labels
    )

    st.divider()

    st.header(
        "Model Settings"
    )

    history_days = st.slider(
        "Historical data window",
        min_value=15,
        max_value=90,
        value=45,
        step=15
    )

    min_sample = st.slider(
        "Minimum games for prediction",
        min_value=3,
        max_value=10,
        value=5
    )


# ============================================================
# LIVE ODDS
# ============================================================

if not key:

    st.warning(
        "Enter your Odds API key in the sidebar."
    )

    st.stop()


try:

    ev = events(key)

except Exception as e:

    st.error(
        str(e)
    )

    st.stop()


if not ev:

    st.warning(
        "No current/upcoming WNBA games were returned."
    )

    st.stop()


options = {}

for e in ev:

    try:

        time = pd.to_datetime(
            e["commence_time"]
        ).strftime(
            "%b %d %I:%M %p"
        )

    except Exception:

        time = ""

    label = (
        f"{e['away_team']} @ "
        f"{e['home_team']} — {time}"
    )

    options[label] = e["id"]


chosen = st.selectbox(
    "WNBA game",
    list(options)
)


try:

    e = event_odds(
        key,
        options[chosen],
        wanted
    )

except Exception as ex:

    st.error(
        str(ex)
    )

    st.stop()


df = flatten(
    e,
    wanted
)


if df.empty:

    st.warning(
        "No selected player props were returned "
        "for this game."
    )

    st.stop()


if selected_books:

    df = df[
        df["bookmaker"].isin(
            selected_books
        )
    ].copy()


if df.empty:

    st.warning(
        "No props remain after applying "
        "your sportsbook filter."
    )

    st.stop()


df["implied_prob"] = df[
    "odds"
].apply(implied)


# ============================================================
# HISTORICAL DATA
# ============================================================

with st.spinner(
    "Loading real WNBA player history..."
):

    historical = build_historical_dataset(
        history_days
    )


if historical.empty:

    st.warning(
        "The live odds loaded, but historical "
        "player data could not be retrieved. "
        "The model cannot produce a prediction yet."
    )

else:

    st.success(
        f"Loaded {len(historical):,} real player-game "
        f"records from the last {history_days} days."
    )


# ============================================================
# RUN MODEL
# ============================================================

model_rows = []

if not historical.empty:

    for _, row in df.iterrows():

        prediction = player_projection(
            historical,
            row["player"],
            row["market_name"],
            row["line"]
        )

        if prediction is None:
            continue

        if prediction["sample_size"] < min_sample:
            continue

        model_prob = (
            prediction["over_probability"]
            if row["side"].lower() == "over"
            else prediction["under_probability"]
        )

        new_row = row.copy()

        new_row[
            "projection"
        ] = prediction["projection"]

        new_row[
            "model_probability"
        ] = model_prob

        new_row[
            "sample_size"
        ] = prediction["sample_size"]

        new_row[
            "last5"
        ] = prediction["last5"]

        new_row[
            "last10"
        ] = prediction["last10"]

        new_row[
            "last20"
        ] = prediction["last20"]

        new_row[
            "minutes"
        ] = prediction["minutes"]

        model_rows.append(
            new_row
        )


model_df = pd.DataFrame(
    model_rows
)


if not model_df.empty:

    scores = model_df.apply(
        score_bet,
        axis=1
    )

    model_df = pd.concat(
        [
            model_df.reset_index(drop=True),
            scores.reset_index(drop=True)
        ],
        axis=1
    )

    model_df = model_df.sort_values(
        "edge",
        ascending=False
    )


# ============================================================
# DASHBOARD
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "WNBA games",
    len(ev)
)

c2.metric(
    "Prop rows",
    len(df)
)

c3.metric(
    "Players",
    df["player"].nunique()
)

c4.metric(
    "Books",
    df["bookmaker"].nunique()
)


# ============================================================
# AI BEST BETS
# ============================================================

st.subheader(
    "🤖 AI Model — Best Bets"
)


if model_df.empty:

    st.warning(
        "No players currently have enough historical "
        "data for a model prediction."
    )

else:

    best = model_df[
        model_df["edge"] >= 0.02
    ].copy()

    if best.empty:

        st.info(
            "The model currently sees no bets with at "
            "least a 2% edge. That's a feature, not a bug."
        )

    else:

        display = best.head(25).copy()

        display["projection"] = display[
            "projection"
        ].map(
            lambda x: f"{x:.2f}"
        )

        display["model_probability"] = display[
            "model_probability"
        ].map(
            lambda x: f"{x:.1%}"
        )

        display["market_probability"] = display[
            "market_probability"
        ].map(
            lambda x: f"{x:.1%}"
        )

        display["edge"] = display[
            "edge"
        ].map(
            lambda x: f"{x:+.1%}"
        )

        display["expected_value"] = display[
            "expected_value"
        ].map(
            lambda x: f"{x:+.1%}"
        )

        display["odds"] = display[
            "odds"
        ].map(
            lambda x: f"{x:+.0f}"
        )

        display = display[
            [
                "rating",
                "player",
                "market_name",
                "side",
                "line",
                "projection",
                "model_probability",
                "market_probability",
                "edge",
                "expected_value",
                "odds",
                "bookmaker",
                "sample_size"
            ]
        ]

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL DETAIL
# ============================================================

if not model_df.empty:

    st.subheader(
        "📈 Model Analysis"
    )

    selected_player = st.selectbox(
        "Select a player to inspect",
        sorted(
            model_df["player"].unique()
        )
    )

    player_bets = model_df[
        model_df["player"]
        == selected_player
    ].copy()

    if not player_bets.empty:

        cols = st.columns(4)

        best_row = player_bets.iloc[0]

        cols[0].metric(
            "Model projection",
            f"{best_row['projection']:.2f}"
        )

        cols[1].metric(
            "Last 5",
            f"{best_row['last5']:.2f}"
        )

        cols[2].metric(
            "Last 10",
            f"{best_row['last10']:.2f}"
        )

        cols[3].metric(
            "Games",
            int(best_row["sample_size"])
        )

        detail = player_bets.copy()

        detail["model_probability"] = detail[
            "model_probability"
        ].map(
            lambda x: f"{x:.1%}"
        )

        detail["edge"] = detail[
            "edge"
        ].map(
            lambda x: f"{x:+.1%}"
        )

        detail["expected_value"] = detail[
            "expected_value"
        ].map(
            lambda x: f"{x:+.1%}"
        )

        detail["odds"] = detail[
            "odds"
        ].map(
            lambda x: f"{x:+.0f}"
        )

        st.dataframe(
            detail[
                [
                    "market_name",
                    "side",
                    "line",
                    "projection",
                    "model_probability",
                    "edge",
                    "expected_value",
                    "odds",
                    "bookmaker",
                    "rating"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MARKET CONSENSUS
# ============================================================

st.subheader(
    "📊 Market Consensus"
)

cons = []

for (
    player,
    market,
    line
), g in df.groupby(
    [
        "player",
        "market_name",
        "line"
    ]
):

    over = g[
        g.side.str.lower()
        == "over"
    ]

    under = g[
        g.side.str.lower()
        == "under"
    ]

    for book in set(
        over.bookmaker
    ) & set(
        under.bookmaker
    ):

        o = over[
            over.bookmaker == book
        ].iloc[0]

        u = under[
            under.bookmaker == book
        ].iloc[0]

        po = implied(
            o.odds
        )

        pu = implied(
            u.odds
        )

        total = po + pu

        if total:

            cons.append(
                [
                    player,
                    market,
                    line,
                    book,
                    po / total,
                    pu / total
                ]
            )


cd = pd.DataFrame(
    cons,
    columns=[
        "player",
        "market",
        "line",
        "book",
        "over_p",
        "under_p"
    ]
)


if cd.empty:

    st.warning(
        "Not enough two-sided prices to calculate "
        "no-vig consensus."
    )

else:

    out = []

    for (
        player,
        market,
        line
    ), g in cd.groupby(
        [
            "player",
            "market",
            "line"
        ]
    ):

        po = g.over_p.mean()
        pu = g.under_p.mean()

        out += [
            [
                player,
                market,
                "Over",
                line,
                po,
                fair_odds(po)
            ],
            [
                player,
                market,
                "Under",
                line,
                pu,
                fair_odds(pu)
            ]
        ]

    consensus = pd.DataFrame(
        out,
        columns=[
            "player",
            "market",
            "side",
            "line",
            "consensus_prob",
            "fair_odds"
        ]
    )

    consensus[
        "consensus_prob"
    ] = consensus[
        "consensus_prob"
    ].map(
        lambda x: f"{x:.1%}"
    )

    consensus[
        "fair_odds"
    ] = consensus[
        "fair_odds"
    ].map(
        lambda x:
        "—"
        if pd.isna(x)
        else f"{x:+.0f}"
    )

    st.dataframe(
        consensus,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RAW ODDS
# ============================================================

with st.expander(
    "🏪 Best Available Prices"
):

    show = df.copy()

    show["implied_prob"] = show[
        "implied_prob"
    ].map(
        lambda x: f"{x:.1%}"
    )

    show["odds"] = show[
        "odds"
    ].map(
        lambda x: f"{x:+.0f}"
    )

    st.dataframe(
        show[
            [
                "player",
                "market_name",
                "side",
                "line",
                "bookmaker",
                "odds",
                "implied_prob"
            ]
        ].sort_values(
            [
                "player",
                "market_name",
                "line",
                "side"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL EXPLANATION
# ============================================================

with st.expander(
    "🧠 How EdgeFinder AI makes predictions"
):

    st.markdown(
        """
### Current model

The prediction engine uses actual WNBA player game data.

For each player/prop it considers:

- Last 5 games
- Last 10 games
- Last 20 games
- Recent minutes
- Overall volatility
- Recent-form weighting

The projection weights recent performance more heavily:

**50% last 5 + 30% last 10 + 20% last 20**

The model then estimates the probability of the player finishing above or below the sportsbook's line.

### Edge calculation

**Model probability − sportsbook implied probability = model edge**

For example:

Model probability: **61%**

Sportsbook implied probability: **54.5%**

Model edge: **+6.5%**

The app then calculates expected value using the actual sportsbook odds.

### Important

This is an experimental statistical model.

A positive model edge does **not** guarantee a winning bet.

The next major upgrade should be backtesting the model against historical games and tracking calibration, ROI, hit rate and closing-line value.
"""
    )


st.caption(
    "EdgeFinder AI • Live WNBA odds + statistical player model • "
    + datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
)
