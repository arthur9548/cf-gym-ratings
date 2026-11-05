
#!/usr/bin/env python3
"""Estimate Codeforces Gym problem ratings with a regularized Elo/Rasch model.

Requires: pip install requests
Credentials: CODEFORCES_API_KEY, CODEFORCES_API_SECRET

Examples:
    python gym_ratings.py 106523
    python gym_ratings.py 106523 --team handle1,handle2,handle3
    python gym_ratings.py 106523 --team-name "Example Team"
    python gym_ratings.py 106523 --team-id 12345
    python gym_ratings.py 105900 --rating-mode current
    python gym_ratings.py 106159 --compare --spread 600

Default: historical blend (75% historical peak) / ghost weight 0.5 / spread 500.
For historical modes, current-rating problem and team previews are printed before
API history fetches. The remaining wait is estimated using uncached histories.

The ratings are heuristic estimates, not official Codeforces ratings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

# API rate limit is one request per two seconds.
API_DELAY = 2.2
RATING_BATCH_SIZE = 50

# Maximum-posterior regularization for unrated teams.
UNRATED_TEAM_SD = 300.0
RATED_TEAM_SD = 180.0  # historical Codeforces ratings are noisy anchors
COHORT_MEAN_SD = 500.0
# One artificial reference team's 0.5 success / 0.5 failure per problem.
ITEM_PSEUDO_COUNT = 1.0

_LAST_API_CALL = -1e20


def api_call(method: str, params: dict[str, str], authenticated: bool = False):
    global _LAST_API_CALL

    params = dict(params)
    if authenticated:
        key = os.environ["CODEFORCES_API_KEY"].strip()
        secret = os.environ["CODEFORCES_API_SECRET"].strip()
        params["apiKey"] = key
        params["time"] = str(int(time.time()))

        # The string signed by Codeforces uses raw sorted parameter values.
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        rand = secrets.token_hex(3)  # exactly six characters
        message = f"{rand}/{method}?{query}#{secret}"
        params["apiSig"] = rand + hashlib.sha512(message.encode()).hexdigest()

    remaining = API_DELAY - (time.monotonic() - _LAST_API_CALL)
    if remaining > 0:
        time.sleep(remaining)
    _LAST_API_CALL = time.monotonic()

    response = requests.get(
        f"https://codeforces.com/api/{method}", params=params, timeout=60
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{method} returned HTTP {response.status_code}: "
            f"{response.text[:250]!r}"
        ) from exc
    if data.get("status") != "OK":
        raise RuntimeError(
            f"{method} (HTTP {response.status_code}): "
            f"{data.get('comment', str(data))}"
        )
    response.raise_for_status()
    return data["result"]


def get_standings(contest_id: str):
    return api_call("contest.standings", {
        "contestId": contest_id,
        "showUnofficial": "true",
        "participantTypes": "CONTESTANT,VIRTUAL,OUT_OF_COMPETITION",
    }, authenticated=True)


def get_user_ratings(handles: list[str]):
    """Return a case-insensitive mapping of handle -> current rating or None."""
    ratings = {}

    def fetch_batch(batch):
        try:
            # The API guarantees response order matches request order.
            users = api_call("user.info", {"handles": ";".join(batch)})
        except RuntimeError as exc:
            # A single deleted/unresolvable account invalidates a batch.
            msg = str(exc).lower()
            if "not found" not in msg and "not exist" not in msg:
                raise
            if len(batch) == 1:
                print(f"Warning: unavailable handle: {batch[0]}")
                ratings[batch[0].casefold()] = None
                return
            half = len(batch) // 2
            fetch_batch(batch[:half])
            fetch_batch(batch[half:])
            return
        if len(users) != len(batch):
            raise RuntimeError("user.info returned an unexpected user count")
        for handle, user in zip(batch, users):
            ratings[handle.casefold()] = user.get("rating")

    for start in range(0, len(handles), RATING_BATCH_SIZE):
        fetch_batch(handles[start : start + RATING_BATCH_SIZE])
    return ratings


def save_history_cache(path: Path, cache: dict):
    """Write atomically so an interrupted run does not destroy the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def format_duration(seconds: float):
    """Human-readable approximate duration, rounded up to 10 seconds."""
    rounded = int(math.ceil(max(0.0, seconds) / 10.0)) * 10
    if rounded < 60:
        return f"{rounded}s"
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m" + (f" {secs}s" if secs else "")
    return f"{minutes}m" + (f" {secs}s" if secs else "")


def get_user_histories(handles: list[str], cache_path: Path, refresh=False,
                       model_seconds: float = 0.0):
    """Return rating histories, with an approximate ETA before slow API requests.

    One user.rating call is needed for each uncached handle. Codeforces rate
    limiting sets the baseline wait; actual network latency may increase it.
    model_seconds is the duration observed for the current-rating preview fit.
    """
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    missing = [h for h in handles if refresh or h.casefold() not in cache]
    print(f"Historical rating histories: {len(handles) - len(missing)} cached, "
          f"{len(missing)} to fetch")

    estimated = len(missing) * API_DELAY + model_seconds
    if missing:
        print(f"Estimated time to final results: ~{format_duration(estimated)} "
              f"({len(missing)} history requests at ~{API_DELAY:g}s/request "
              "plus model fitting; network delays may vary)", flush=True)
    else:
        print(f"All rating histories are cached. Final model fit: "
              f"~{format_duration(model_seconds)}.", flush=True)

    started = time.monotonic()
    for i, handle in enumerate(missing, 1):
        try:
            history = api_call("user.rating", {"handle": handle})
        except RuntimeError as exc:
            # Invalid/deleted handles should not abort the entire contest.
            msg = str(exc).lower()
            if "not found" not in msg and "not exist" not in msg:
                raise
            print(f"Warning: no rating history for unavailable handle {handle}")
            history = []

        cache[handle.casefold()] = sorted(
            history, key=lambda change: change["ratingUpdateTimeSeconds"]
        )
        if i % 10 == 0 or i == len(missing):
            save_history_cache(cache_path, cache)
            # Update the ETA using the observed request throughput while
            # maintaining the Codeforces pacing as a lower bound.
            elapsed = time.monotonic() - started
            seconds_per_call = max(API_DELAY, elapsed / i)
            remaining = (len(missing) - i) * seconds_per_call + model_seconds
            print(f"  Rating histories fetched: {i}/{len(missing)}; "
                  f"estimated time remaining: ~{format_duration(remaining)}",
                  flush=True)

    return {handle.casefold(): cache.get(handle.casefold(), []) for handle in handles}


def historical_member_rating(history: list[dict], timestamp: int,
                             mode: str, peak_weight: float):
    """Compute rating strictly from updates published before participation."""
    past = [change["newRating"] for change in history
            if change["ratingUpdateTimeSeconds"] < timestamp]
    if not past:
        return None

    at_time = past[-1]
    peak = max(past)
    if mode == "at-time":
        return float(at_time)
    if mode == "peak-at-time":
        return float(peak)
    if mode == "blend":
        return (1 - peak_weight) * at_time + peak_weight * peak
    raise ValueError(f"Unsupported historical rating mode: {mode}")


def row_participation_time(party: dict, contest: dict):
    """Avoid dating later virtual/OOC participation with the Gym's start."""
    if party.get("startTimeSeconds") is not None:
        return party["startTimeSeconds"]
    if party.get("participantType") == "CONTESTANT":
        return contest.get("startTimeSeconds")
    # No reliable timestamp for virtual/OOC rows without startTimeSeconds.
    return None


def elo_probability(strength: float, difficulty: float, spread: float):
    """Probability that a participant solves a problem, without overflow."""
    exponent = (strength - difficulty) / spread
    if exponent >= 0:
        return 1.0 / (1.0 + 10.0 ** (-exponent))
    t = 10.0 ** exponent
    return t / (1.0 + t)


def codeforces_team_rating(member_ratings: list[float]):
    """Mike Mirzayanov's team-rating aggregation (Codeforces blog 16986)."""
    if len(member_ratings) == 1:
        return float(member_ratings[0])
    lo = min(member_ratings) - 1000.0
    hi = max(member_ratings) + 3000.0
    for _ in range(55):
        mid = (lo + hi) / 2.0
        p_opponent_beats_team = math.prod(
            elo_probability(mid, r, 400.0) for r in member_ratings
        )
        if p_opponent_beats_team < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class Team:
    name: str
    outcomes: tuple[int, ...]
    cohort: str  # 'rated', 'ghost', or 'unrated_online'
    weight: float
    rating: Optional[float] = None
    strength: float = 0.0
    handles: tuple[str, ...] = ()
    team_id: Optional[int] = None
    participant_type: str = ""
    points: float = 0.0
    penalty: int = 0
    api_rank: Optional[int] = None
    row_number: int = -1


def parse_teams(rows, n_problems, user_ratings, ghost_weight, ooc_weight,
                contest, rating_mode, peak_weight):
    teams = []
    inactive = 0
    missing_times = 0
    missing_history = 0
    for row_number, row in enumerate(rows, 1):
        party = row["party"]
        participant_type = party["participantType"]
        if participant_type not in {"CONTESTANT", "VIRTUAL", "OUT_OF_COMPETITION"}:
            continue
        results = row["problemResults"]
        if len(results) != n_problems:
            continue
        outcomes = tuple(int(p.get("points", 0) > 0) for p in results)

        # No accepted solutions AND no rejected submissions: likely a no-show.
        if not any(outcomes) and not any(
            p.get("rejectedAttemptCount", 0) > 0 for p in results
        ):
            inactive += 1
            continue

        members = party.get("members", [])
        name = party.get("teamName") or ", ".join(
            m.get("handle", "?") for m in members
        ) or "(unnamed team)"
        details = dict(
            handles=tuple(m["handle"] for m in members if m.get("handle")),
            team_id=party.get("teamId"),
            participant_type=participant_type,
            points=float(row.get("points", sum(outcomes))),
            penalty=int(row.get("penalty", 0)),
            api_rank=row.get("rank"),
            row_number=row_number,
        )
        if party.get("ghost", False):
            teams.append(Team(name, outcomes, "ghost", ghost_weight, **details))
            continue

        weight = ooc_weight if participant_type == "OUT_OF_COMPETITION" else 1.0
        timestamp = (row_participation_time(party, contest)
                     if rating_mode != "current" else None)

        ratings = []
        for member in members:
            handle = member.get("handle", "").casefold()
            if rating_mode == "current":
                rating = user_ratings.get(handle)
            elif timestamp is None:
                rating = None
            else:
                rating = historical_member_rating(
                    user_ratings.get(handle, []), timestamp, rating_mode, peak_weight
                )
            ratings.append(rating)

        if ratings and all(r is not None for r in ratings):
            rating = codeforces_team_rating(ratings)
            teams.append(Team(name, outcomes, "rated", weight, rating, rating,
                              **details))
        else:
            teams.append(Team(name, outcomes, "unrated_online", weight,
                              **details))
            if rating_mode != "current":
                if timestamp is None:
                    missing_times += 1
                else:
                    missing_history += 1

    return teams, inactive, missing_times, missing_history


def fit_model(teams: list[Team], n_problems: int, spread: float):
    """Coordinate-ascent MAP with soft rated-team anchors and latent team strengths.

    Objective: weighted Bernoulli log-likelihood minus normal priors on
    rated and unrated strengths/cohort means, plus one Jeffreys-style fractional
    pseudo-observation per problem at the rated median strength.
    """
    anchors = [t.rating for t in teams if t.cohort == "rated" and t.weight > 0]
    if not anchors:
        return None

    reference = statistics.median(anchors)
    scale = spread / math.log(10.0)
    cohorts = {"ghost": reference, "unrated_online": reference}
    for team in teams:
        team.strength = team.rating if team.cohort == "rated" else reference

    def difficulty_root(j):
        # The pseudo participant makes even zero/all-solved items finite.
        observed = sum(t.weight * t.outcomes[j] for t in teams)
        target = observed + ITEM_PSEUDO_COUNT / 2
        lo, hi = -1000.0, 6000.0
        for _ in range(39):
            d = (lo + hi) / 2
            expected = sum(
                t.weight * elo_probability(t.strength, d, spread)
                for t in teams
            ) + ITEM_PSEUDO_COUNT * elo_probability(reference, d, spread)
            if expected > target:
                lo = d
            else:
                hi = d
        return (lo + hi) / 2

    def strength_root(team, difficulties, prior_mean, sd):
        # Solve: weight*sum(p-y) + scale*(theta-mu)/SD^2 = 0.
        prior_factor = scale / (sd * sd)
        lo, hi = -1000.0, 6000.0
        actual = sum(team.outcomes)
        for _ in range(39):
            r = (lo + hi) / 2
            expected = sum(elo_probability(r, d, spread) for d in difficulties)
            root = team.weight * (expected - actual) + prior_factor * (r - prior_mean)
            if root > 0:
                hi = r
            else:
                lo = r
        return (lo + hi) / 2

    difficulties = [reference] * n_problems
    iterations = 0
    for iteration in range(80):
        iterations = iteration + 1
        previous = difficulties[:]
        difficulties = [difficulty_root(j) for j in range(n_problems)]

        for team in teams:
            if team.cohort == "rated":
                # Team rating anchors the scale, but observed performance may
                # update individual strength within a regularized margin.
                team.strength = strength_root(
                    team, difficulties, team.rating, RATED_TEAM_SD
                )
            else:
                team.strength = strength_root(
                    team, difficulties, cohorts[team.cohort], UNRATED_TEAM_SD
                )

        previous_cohorts = dict(cohorts)
        shrink = (UNRATED_TEAM_SD / COHORT_MEAN_SD) ** 2
        for cohort in cohorts:
            members = [t.strength for t in teams if t.cohort == cohort]
            if members:
                cohorts[cohort] = (sum(members) + shrink * reference) / (
                    len(members) + shrink
                )

        change = max(
            max(abs(a - b) for a, b in zip(difficulties, previous)),
            max(abs(cohorts[c] - previous_cohorts[c]) for c in cohorts),
        )
        if change < 0.05:
            break

    # Once more with the final strengths and cohort locations.
    difficulties = [difficulty_root(j) for j in range(n_problems)]
    return difficulties, reference, cohorts, iterations


def format_fraction(group: list[Team], problem_index: int):
    if not group:
        return "-"
    count = sum(t.outcomes[problem_index] for t in group)
    return f"{count}/{len(group)} ({100.0 * count / len(group):.1f}%)"


def display_estimates(contest_id, problems, teams, ratings):
    print("\nProblem ratings (rounded to nearest 100; heuristic):")
    print(
        f"{'Problem':<8} {'Estimated Rating':>16}  {'Rated Online':>22}  "
        f"{'Ghosts':>22}  {'All Participants':>22}  Name"
    )
    rated = [t for t in teams if t.cohort == "rated"]
    ghosts = [t for t in teams if t.cohort == "ghost"]

    for j, problem in enumerate(problems):
        rounded = round(ratings[j] / 100) * 100
        displayed = max(800, min(4000, rounded))
        marker = "*" if ratings[j] < 800 or ratings[j] > 4000 else " "
        successful = sum(t.outcomes[j] for t in teams)
        if successful == 0 or successful == len(teams):
            marker = "?"
        print(
            f"{problem['index']:<8} {str(displayed) + marker:>16}  "
            f"{format_fraction(rated, j):>22}  "
            f"{format_fraction(ghosts, j):>22}  "
            f"{format_fraction(teams, j):>22}  {problem['name']}"
        )
    print("  ? = zero/all solves in usable rows; estimate is highly uncertain")
    print("  * = raw estimate outside the display range [800, 4000]")
    print(f"\nProblems: https://codeforces.com/gym/{contest_id}/problems")


def team_description(team: Team):
    extra = ", ".join(team.handles)
    return f"{team.name} ({extra})" if extra and extra != team.name else team.name


def choose_team(teams: list[Team], args):
    """Match by one or more member handles, Codeforces team ID, or team name."""
    if args.team is not None:
        handles = {h.strip().casefold() for h in args.team.split(",") if h.strip()}
        if not handles:
            raise ValueError("--team requires comma-separated Codeforces handles")
        matches = [t for t in teams if handles.issubset({h.casefold() for h in t.handles})]
    elif args.team_id is not None:
        matches = [t for t in teams if t.team_id == args.team_id]
    elif args.team_name is not None:
        name = args.team_name.strip().casefold()
        if not name:
            raise ValueError("--team-name must be nonempty")
        exact = [t for t in teams if t.name.casefold() == name]
        matches = exact or [t for t in teams if name in t.name.casefold()]
    else:
        return None

    if not matches:
        raise ValueError("No active modeled team matched that selection. "
                         "Check the handles, team name or ID; no-shows are excluded.")
    if len(matches) > 1:
        candidates = "\n".join(
            f"  row {t.row_number}: {team_description(t)} "
            f"[teamId={t.team_id}, type={t.participant_type}]"
            for t in matches[:12]
        )
        raise ValueError(
            f"Team selection is ambiguous ({len(matches)} matches). "
            f"Use --team-id or a more specific --team-name if possible:\n{candidates}"
        )
    return matches[0]


def original_strength(team: Team):
    """Use pre-event rating anchors; only infer strength for unrated teams."""
    return team.rating if team.rating is not None else team.strength


def expected_place(rating: float, opponents: list[float]):
    """Seed under Codeforces-style 400-point pairwise win probabilities."""
    return 1.0 + sum(elo_probability(opp, rating, 400.0) for opp in opponents)


def placement_performance(midrank: float, opponents: list[float]):
    """Invert expected placement, smoothing boundary ranks to stay finite.

    The half-position correction maps first/last places strictly inside the
    attainable [1, n] expected-rank interval. This is not an official CF formula.
    """
    n = len(opponents) + 1
    if n < 2:
        return None
    adjusted_rank = 1.0 + (midrank - 0.5) * (n - 1.0) / n
    lo = min(opponents) - 5000.0
    hi = max(opponents) + 5000.0
    for _ in range(75):
        mid = (lo + hi) / 2
        if expected_place(mid, opponents) > adjusted_rank:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def display_team_analysis(team: Team, teams: list[Team], rating_mode: str,
                          stage: str = "final model"):
    """Compare actual ICPC standing against expected Codeforces-like seed.

    The comparison population is all usable rows in the model. Standings from
    original and virtual contests were not contemporaneous, so the ranking is
    a useful hypothetical comparison rather than an official Codeforces rank.
    """
    n = len(teams)
    score = (team.points, -team.penalty)
    rank = 1 + sum(
        (other.points, -other.penalty) > score
        for other in teams if other is not team
    )
    ties = sum((t.points, -t.penalty) == score for t in teams)
    midrank = rank + (ties - 1) / 2.0

    team_rating = original_strength(team)
    opponents = [original_strength(other) for other in teams if other is not team]
    seed = expected_place(team_rating, opponents)
    performance = placement_performance(midrank, opponents)

    print(f"\nTeam analysis ({stage}):")
    print(f"  Team: {team.name}")
    if team.handles:
        print(f"  Handles: {', '.join(team.handles)}")
    if team.team_id is not None:
        print(f"  Team ID: {team.team_id}")
    print(f"  Participant type: {team.participant_type}" +
          (" (ghost)" if team.cohort == "ghost" else ""))
    print(f"  Solved: {sum(team.outcomes)}/{len(team.outcomes)}; "
          f"penalty: {team.penalty}")
    if team.rating is not None:
        print(f"  Team rating ({rating_mode} members): {team_rating:.0f}")
    else:
        print(f"  Estimated team strength (fitted from this Gym): {team_rating:.0f}")

    suffix = f" (tied with {ties - 1} other team(s))" if ties > 1 else ""
    print(f"  Actual placement (modeled field): {rank}/{n}{suffix}")
    if team.api_rank is not None:
        print(f"  Codeforces-reported rank: {team.api_rank}")
    print(f"  Expected placement (model): {seed:.1f}/{n}")
    if performance is not None:
        print(f"  Rank-equivalent performance: {performance:.0f}")
        print(f"  Performance minus team rating: {performance - team_rating:+.0f}")
    else:
        print("  Rank-equivalent performance: N/A (only one team)")
    print("  Note: these are modeled rankings across original and online rows, "
          "not official CF performance ratings.")
    if team.rating is None:
        print("  Note: this team's strength was inferred from these same results; "
              "its expected rank is not an independent prediction.")
    else:
        print("  Note: unrated opponents are fitted from the same Gym outcomes, "
              "so the expected rank is not fully independent of these results.")


def print_model_summary(contest_id, contest, problems, teams, inactive,
                        missing_times, missing_history, args, rating_mode,
                        heading):
    counts = {c: sum(t.cohort == c for t in teams)
              for c in ("rated", "ghost", "unrated_online")}
    print(f"\n{'=' * 18} {heading} {'=' * 18}")
    print(f"Contest: {contest['name']} ({contest_id})")
    print(f"Member rating mode: {rating_mode}")
    print(f"Usable standings rows: {len(teams)} "
          f"(excluded likely no-shows: {inactive})")
    if rating_mode != "current":
        print(f"Rows without reliable start times: {missing_times}; "
              f"rows with insufficient historical ratings: {missing_history}")
    print(f"Rated online teams: {counts['rated']}; "
          f"unrated online: {counts['unrated_online']}; ghosts: {counts['ghost']}")
    if counts['rated'] < 10:
        print("WARNING: Very few rated teams; absolute calibration may be unstable.")
    if not counts['rated']:
        print("No rated online teams: absolute calibration unavailable.")
        print("Problem solve counts (not calibrated ratings):")
        for j, p in enumerate(problems):
            s = sum(t.outcomes[j] for t in teams)
            print(f"  {p['index']}: {s}/{len(teams)} solved")
        return None

    outcome = fit_model(teams, len(problems), args.spread)
    difficulty, reference, cohorts, iterations = outcome
    print(f"Rated-team median: {reference:.0f}; model iterations: {iterations}")
    if iterations == 80:
        print("WARNING: Fit reached its iteration limit; check sensitivity.")
    if counts['ghost']:
        print(f"Inferred ghost cohort strength: {cohorts['ghost']:.0f}")
    if counts['unrated_online']:
        print(f"Inferred unrated-online cohort strength: {cohorts['unrated_online']:.0f}")
    print(f"Ghost weight: {args.ghost_weight}; probability spread: {args.spread}")
    if rating_mode == "blend":
        print(f"Historical member rating: {args.peak_weight:.0%} peak + "
              f"{1 - args.peak_weight:.0%} at-time")

    display_estimates(contest_id, problems, teams, difficulty)
    sys.stdout.flush()  # Display the preview before slow historical API requests.
    return difficulty


def run(args):
    contest_id = args.contest_id or input("contestid: ").strip()
    result = get_standings(contest_id)
    contest = result["contest"]
    problems = result["problems"]
    rows = result["rows"]

    if contest["type"] != "ICPC":
        raise RuntimeError("This script supports only binary ICPC scoring.")

    handles = sorted({
        m["handle"]
        for row in rows
        if not row["party"].get("ghost", False)
        for m in row["party"].get("members", [])
        if m.get("handle")
    }, key=str.casefold)

    # Stage 1: fast current-rating preview. When current is selected, it is final.
    print(f"Fetching current ratings for {len(handles)} unique handles...")
    current_ratings = get_user_ratings(handles)
    current_teams, inactive, _, _ = parse_teams(
        rows, len(problems), current_ratings, args.ghost_weight, args.ooc_weight,
        contest, "current", args.peak_weight
    )
    current_label = "FINAL — current ratings" if args.rating_mode == "current" else \
                    "PREVIEW — current ratings"
    preview_start = time.monotonic()
    current_difficulty = print_model_summary(
        contest_id, contest, problems, current_teams, inactive,
        0, 0, args, "current", current_label
    )
    preview_model_seconds = time.monotonic() - preview_start

    wants_team = any(x is not None for x in
                     (args.team, args.team_id, args.team_name))
    if wants_team:
        # Resolve the selected team before slow historical API calls; also
        # provide its performance as part of the fast current-rating preview.
        current_team = choose_team(current_teams, args)
        if current_difficulty is not None:
            stage = ("final model" if args.rating_mode == "current" else
                     "preview — current ratings")
            display_team_analysis(current_team, current_teams, "current", stage)
        else:
            print("Team performance preview unavailable without calibrated "
                  "opponent ratings.")
        sys.stdout.flush()

    if args.rating_mode == "current":
        teams, difficulty = current_teams, current_difficulty
    else:
        # Stage 2: fetch only uncached histories; same contest and model settings.
        print("\nFetching historical rating data for final estimates...", flush=True)
        histories = get_user_histories(
            handles, Path(args.rating_cache).expanduser(), args.refresh_ratings,
            model_seconds=preview_model_seconds
        )
        teams, inactive, missing_times, missing_history = parse_teams(
            rows, len(problems), histories, args.ghost_weight, args.ooc_weight,
            contest, args.rating_mode, args.peak_weight
        )
        difficulty = print_model_summary(
            contest_id, contest, problems, teams, inactive,
            missing_times, missing_history, args, args.rating_mode,
            "FINAL — historical ratings"
        )

    if difficulty is None:
        if wants_team and args.rating_mode != "current":
            print("Final team performance unavailable without calibrated "
                  "opponent ratings.")
        return

    # The selected team was already printed above in current mode.
    if wants_team and args.rating_mode != "current":
        selected = choose_team(teams, args)
        display_team_analysis(selected, teams, args.rating_mode)

    if args.compare and any(t.cohort == "ghost" for t in teams):
        initial_weights = [t.weight for t in teams]
        baseline_results = {}
        for weight in (0.0, 1.0):
            for t, initial in zip(teams, initial_weights):
                t.weight = weight if t.cohort == "ghost" else initial
            baseline_results[weight] = fit_model(teams, len(problems), args.spread)[0]
        for t, initial in zip(teams, initial_weights):
            t.weight = initial
        print("\nSensitivity to ghost weight (unrounded ratings):")
        print(" Problem    0 (none)   chosen   1 (equal)")
        for j, p in enumerate(problems):
            print(f"    {p['index']:>2}     {baseline_results[0.0][j]:>7.0f}  "
                  f"{difficulty[j]:>7.0f}   {baseline_results[1.0][j]:>7.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contest_id", nargs="?", help="Codeforces Gym contest ID")
    parser.add_argument("--ghost-weight", type=float, default=0.5,
                        help="Relative weight for ghost results (default: 0.5)")
    parser.add_argument("--ooc-weight", type=float, default=0.5,
                        help="Weight for out-of-competition online teams (default: 0.5)")
    parser.add_argument("--spread", type=float, default=500.0,
                        help="Elo log-odds rating interval (default: 500)")
    parser.add_argument("--rating-mode", choices=("current", "at-time", "peak-at-time", "blend"),
                        default="blend", help="Member rating prior (default: blend)")
    parser.add_argument("--peak-weight", type=float, default=0.75,
                        help="Peak weight for blend mode (default: 0.75)")
    parser.add_argument("--rating-cache", default=str(
                        Path(__file__).resolve().with_name(".cf_rating_history_cache.json")),
                        help="JSON cache of Codeforces rating histories")
    parser.add_argument("--refresh-ratings", action="store_true",
                        help="Refresh cached historical rating data")
    parser.add_argument("--compare", action="store_true",
                        help="Compare ghost weights 0, chosen, and 1")
    team_selector = parser.add_mutually_exclusive_group()
    team_selector.add_argument("--team", metavar="HANDLES",
                               help="Find a team containing these comma-separated handles")
    team_selector.add_argument("--team-id", type=int, metavar="ID",
                               help="Find a team by its Codeforces team ID")
    team_selector.add_argument("--team-name", metavar="NAME",
                               help="Find a team by its name (case insensitive)")
    args = parser.parse_args()
    if args.ghost_weight < 0 or args.ooc_weight < 0 or args.spread <= 0:
        parser.error("Weights must be nonnegative and spread must be positive")
    if not 0 <= args.peak_weight <= 1:
        parser.error("--peak-weight must lie between 0 and 1")
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))
