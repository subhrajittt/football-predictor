# Football Match Predictor

A machine learning pipeline that predicts football match outcomes (home win / draw / away win) across five major European leagues, using historical results, team form, and Elo ratings. This is the first of a few football analytics projects I'm building. This one covers match prediction; a player performance tracker and a live sentiment analyzer are separate builds.

## Tech Stack

Python, pandas, scikit-learn, Streamlit, API-Football

## Leagues Covered

Premier League, La Liga, Serie A, Bundesliga, Ligue 1. Six seasons each (2020-21 through 2025-26), roughly 10,700 matches total.

Each league is modeled separately, with its own Elo pool and its own trained model. This was a deliberate choice, not an oversight. Teams across different leagues never play each other in normal competition, so there's no real data to calibrate one league's strength scale against another's. Mixing them into one global model would just create noise, not signal.

## How It Works

For every match, the pipeline builds three sets of features. All of them are calculated so that a match never has access to information from its own result or from matches after it, avoiding data leakage:

- **Rolling form** - each team's average points and goals from their last 5 matches
- **Elo ratings** - a continuously updating team strength score per league, starting at 1500 and shifting after every match based on the result and how surprising it was
- **Head-to-head history** - how often the home team has beaten this specific opponent in past meetings, as of that point in time

These feed into a logistic regression model, one per league, which outputs win, draw, and loss probabilities for any matchup.

## Results

Evaluated on a time-based split (train on earlier seasons, test on 2024-25 and 2025-26), so the model is never predicting the past using future information.

| League | Naive Baseline | Model Accuracy |
|---|---|---|
| Premier League | 41.7% | 50.8% |
| La Liga | 46.7% | 53.0% |
| Serie A | 39.3% | 53.2% |
| Bundesliga | 41.2% | 52.1% |
| Ligue 1 | 46.4% | 52.1% |

The naive baseline here is always predicting the most common outcome (usually home win). A model that can't beat that isn't learning anything real. Every league's model does, by 5 to 14 points depending on the league.

**Draws are hard to predict**, consistently across all five leagues. Precision and recall on draws are weak, sometimes close to zero. This is a well-documented limitation in football prediction generally, not a bug: draws don't carry as strong a statistical signal as clear favourite/underdog situations. I tried forcing the model to pay more attention to draws using `class_weight='balanced'`, which did produce more draw predictions but dropped overall accuracy noticeably (Premier League went from 50.7% to 46.7%). I kept the standard model as the real one, since the trade-off wasn't worth it, but the balanced version is still in the code for comparison.

## Probability Calibration

While testing the app, I noticed something worth checking properly. Predictions for clearly strong teams, like Arsenal vs Chelsea, looked overconfident: 70%+ win probabilities for matchups between two genuinely competitive clubs, which didn't match how bookmakers or pundits would actually price it.

I built a calibration check to test this properly rather than go on a gut feeling. The idea is to bucket every prediction by confidence level, then check whether the model's stated confidence actually matched its real accuracy in that bucket. The result confirmed the instinct. In Premier League specifically, the model's most confident bucket (80 to 100%) was right only 73% of the time, a 12 point gap. Other leagues showed smaller but similar patterns.

The likely cause is that Elo, recent form, and head-to-head are correlated features that often point the same direction, and logistic regression doesn't know that. It treats each one as independent evidence, which compounds confidence more than it should.

The fix was wrapping the model in scikit-learn's `CalibratedClassifierCV`, which rescales output probabilities to better match real-world accuracy. Re-running the same calibration check afterward, the overconfidence mostly disappeared. The Premier League's top bucket, for example, went from a 12 point gap to under 2 points. Arsenal vs Chelsea's predicted probability dropped from 74.3% to 66.8%, which reads as far more realistic.

## Data Source Validation

To check whether the pipeline could work with live data instead of manually downloaded CSVs, I integrated the API-Football API and pulled the 2023-24 season through it for all five leagues, then compared every match against the same seasons in the CSV data.

The first pass showed uneven match rates. Premier League and Serie A matched fully, but La Liga, Bundesliga, and Ligue 1 came back with significant gaps. Digging into the unmatched rows, most turned out to be team-naming differences between the two sources (for example, the API returns "Atletico Madrid," the CSV source uses "Ath Madrid"), which I fixed with a name mapping. After that, Bundesliga still had a large gap, and the cause turned out to be different: the API's fixture list for that league included teams that don't actually belong to the top flight in that season, most likely relegated or promoted teams from an adjacent division, such as Fortuna Dusseldorf appearing in Bundesliga results.

The fix was filtering API results down to only the teams already known to exist in that league from the CSV data. After both fixes, all five leagues matched 100% between the two sources, with zero result discrepancies:

| League | Matched |
|---|---|
| Premier League | 380 / 380 |
| La Liga | 342 / 342 |
| Serie A | 380 / 380 |
| Bundesliga | 210 / 210 |
| Ligue 1 | 272 / 272 |

The free tier of API-Football only covers the 2022-2024 seasons. Current-season data requires a paid plan ($19+/month), which I decided wasn't worth paying for on a learning project. The integration code (`api_client.py`) works and is ready to use with a different tier. Swapping the API key is the only change needed to pull live data.

## Features I Considered and Rejected

**Squad market value**, as a way to account for transfers. The idea was that a team that sold its best player and bought a replacement looks identical to Elo and form until enough matches happen to reflect it. I looked at pulling Transfermarkt squad values as a proxy, but decided against it. Transfer fees don't reliably predict on-pitch performance (plenty of expensive signings have flopped, plenty of cheap ones have overperformed), and a proper version of this feature would need squad values calculated per season rather than one current snapshot applied across six years of historical data. Elo and form are derived directly from match outcomes, which is a more honest signal than what a club spent in the transfer market.

## Known Limitations

- **No injury or transfer awareness.** The model has no idea if a key player is missing or just arrived. See above for why this is genuinely difficult to add properly.
- **Draws are underpredicted** across all five leagues.
- **Live updates are manual.** `update_data.py` rebuilds the dataset and retrains from whatever CSVs are sitting in `data/raw/`, but getting new results in still means downloading and dropping in a fresh CSV. Full automation would need a paid API tier and a scheduler.
- **Champions League isn't included.** This was part of the original plan, but it's a genuinely different, harder problem than adding another domestic league. Elo and form built from domestic matches alone don't capture cross-league dynamics well, and comparing team strength across 30+ different leagues and competition levels isn't something six seasons of domestic data can really support. Planned as a future phase, not attempted here.
- **Domestic cups** like the FA Cup and Copa del Rey aren't included yet, mainly for scope reasons. A reasonable next addition, since the data source and pipeline already support it structurally.

## Project Structure

football-predictor/
data/
raw/ season CSVs from football-data.co.uk, all 5 leagues
raw_api/ data pulled via API-Football, used for validation
processed/ final feature-engineered dataset
src/
data_prep.py loads and combines raw seasons, tags league and season
features.py rolling form features
elo.py per-league Elo rating system
head_to_head.py head-to-head win rate
build_dataset.py combines everything into one dataset
train.py trains, evaluates, and calibrates models per league
predict.py predicts arbitrary matchups using calibrated models
calibration.py checks whether predicted confidence matches real accuracy
update_data.py rebuilds dataset and retrains from raw/
api_client.py API-Football integration
compare_sources.py validates API data against CSV data
app.py Streamlit app, league and team selectors, live predictions
requirements.txt

## Running It

```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt

python src/build_dataset.py     # builds the dataset
python src/train.py              # trains and evaluates models, all 5 leagues
python src/calibration.py        # checks probability calibration
streamlit run app.py             # launches the app
```

To use the API-Football integration, create a `.env` file in the project root:

API_FOOTBALL_KEY=your_key_here

## What's Next

- Domestic cup competitions like the FA Cup, Copa del Rey, DFB-Pokal, and Coupe de France
- Champions League, once there's a sensible way to handle cross-league strength comparison
- Automated live updates, if a paid API tier becomes worth it