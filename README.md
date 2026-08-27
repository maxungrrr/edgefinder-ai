# EdgeFinder AI — WNBA v2

Live WNBA player-prop market scanner using The Odds API.

## Replace in GitHub
Replace the existing root files:
- `streamlit_app.py`
- `requirements.txt`
- `README.md`

Commit the changes. Streamlit Community Cloud watches the GitHub repository and normally updates the deployed app automatically.

## Live mode
Enter a free The Odds API key in the app. WNBA current odds and player props are available through the API. Player props are queried one game at a time.

## Markets
Points, 1Q Points, Rebounds, 1Q Rebounds, Assists, 1Q Assists, Threes, PRA, PR, PA and RA.

## Important
This is a market-data release, not the predictive AI yet. It deliberately does not fabricate model probabilities. The next release will add player-performance modeling and out-of-sample paper tracking.
