# Codeforces Gym Rating Estimator

A Python tool for estimating problem difficulty ratings and team performances in Codeforces ICPC-style Gyms.

It uses Codeforces participant ratings, historical rating data, and Gym standings to generate approximate Codeforces-equivalent ratings.

## Setup

Requires **Python 3.10+** and `requests`.

Install the dependency:

```bash
pip install requests
```

Generate an API key and secret at [Codeforces API Settings](https://codeforces.com/settings/api).

Set the credentials:

```bash
export CODEFORCES_API_KEY="your_key"
export CODEFORCES_API_SECRET="your_secret"
```

Save the script as `gym_ratings.py`.

## Basic usage

Estimate ratings for a Gym:

```bash
python3 gym_ratings.py 106523
```

The script first prints a fast preview using current Codeforces ratings, followed by the final estimates using historical ratings.

It displays an approximate completion time and progress while historical ratings are retrieved.

## Team performance

Analyze a team by one or more member handles:

```bash
python3 gym_ratings.py 106523 --team handle1,handle2
```

Alternatively, select a team by name or ID:

```bash
python3 gym_ratings.py 106523 --team-name "Example Team"
python3 gym_ratings.py 106523 --team-id 12345
```

Team analysis displays:

- Combined team rating;
- Solved problems and penalty;
- Actual and expected placement;
- Rank-equivalent performance rating;
- Performance difference relative to team rating.

The fast preview includes team performance, so you do not need to wait for historical ratings to finish before seeing preliminary results.

## Configuration

The default parameters are:

| Parameter | Default |
|---|---|
| `--rating-mode` | `blend` |
| `--peak-weight` | `0.75` |
| `--ghost-weight` | `0.5` |
| `--ooc-weight` | `0.5` |
| `--spread` | `500` |

These can be overridden independently.

For example:

```bash
python3 gym_ratings.py 106523 \
  --rating-mode peak-at-time \
  --ghost-weight 0.25 \
  --spread 600
```

### Rating modes

| Mode | Description |
|---|---|
| `current` | Uses today's Codeforces ratings; fastest |
| `at-time` | Uses ratings at the time of participation |
| `peak-at-time` | Uses maximum ratings achieved before participation |
| `blend` | Combines at-time and historical peak ratings (default) |

For a quick result without fetching historical data:

```bash
python3 gym_ratings.py 106523 --rating-mode current
```

### Other options

Compare results under different ghost weights:

```bash
python3 gym_ratings.py 106523 --compare
```

Force historical rating data to be refreshed:

```bash
python3 gym_ratings.py 106523 --refresh-ratings
```

Show all available options:

```bash
python3 gym_ratings.py --help
```

## Caching and execution time

Historical rating histories are cached locally in:

`.cf_rating_history_cache.json`

The script automatically reuses cached information across Gyms.

The first execution for a Gym with many previously unseen users may require several minutes because Codeforces limits API request frequency. Subsequent runs are generally faster.

Current-rating mode uses batched API requests and does not retrieve historical rating data.

## Output

For each problem, the script prints:

- **Estimated Rating:** approximate problem difficulty, rounded to the nearest 100;
- **Rated Online:** solve count and percentage among online teams with known ratings;
- **Ghosts:** solve count and percentage among original ghost teams;
- **All Participants:** solve count and percentage across all usable standings rows.

An asterisk (`*`) marks ratings outside the displayed 800–4000 range. A question mark (`?`) marks problems with zero or all solves, whose estimates are particularly uncertain.

**Note:** All estimated ratings and performances are unofficial and should be treated as approximations.

For the mathematical methodology and API details, see `TECHNICAL.md`.
