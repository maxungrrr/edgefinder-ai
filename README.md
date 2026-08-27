
# EdgeFinder AI — Sports Betting MVP

A free Streamlit dashboard for researching sports-betting edges.

## Current MVP

- NBA / NFL / MLB selector
- Demo mode with clearly labeled synthetic data
- Optional live odds loading
- American-odds → implied probability
- Model probability → edge %
- Expected value
- Candidate-bet ranking
- Sportsbook filtering
- No automatic bet placement

## Important

The first release deliberately does **not** invent model probabilities for live
odds. The next phase is the actual predictive model, trained on historical
player/game data and backtested out of sample.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy free

1. Create a GitHub repository.
2. Upload `streamlit_app.py`, `requirements.txt`, and this README.
3. Sign into Streamlit Community Cloud with GitHub.
4. Click **Create app**.
5. Select your repository and `streamlit_app.py`.
6. Deploy.

## Live Odds

The app can accept a The Odds API key through the sidebar. For production,
store the key as a Streamlit secret rather than putting it in source code.

## Roadmap

### Phase 2 — NBA model
- historical player game logs
- minutes projection
- usage
- opponent matchup
- pace
- injuries
- home/away
- rest
- rolling averages
- XGBoost / gradient boosting
- probability calibration
- out-of-sample backtesting

### Phase 3
- NFL player props
- MLB player props
- line movement
- closing-line value
- bankroll tracking
- model-vs-market diagnostics

### Phase 4
- automated daily scan
- alerts
- model ensembles
- separate models by market
- paper-trading ledger
