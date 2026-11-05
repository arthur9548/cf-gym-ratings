# Codeforces Gym Rating Estimator — Technical Documentation

## 1. Overview

The Codeforces Gym Rating Estimator assigns approximate Codeforces-equivalent difficulty ratings to problems in ICPC-style Gym contests. It also estimates team ratings, expected placements, and contest performance ratings.

Unlike simple solve-rate estimators, the model accounts for differences in participant strength. It combines:

- Codeforces ratings of online participants;
- Historical rating information when available;
- Results of unrated online participants;
- Results of ghost teams representing original on-site contestants.

The underlying statistical framework is a **regularized Rasch model**, also known as one-parameter Item Response Theory (1PL IRT), with Elo-style solve probabilities.

The estimated problem difficulties and team performances are **not official Codeforces ratings**. They are heuristic estimates calibrated to the Codeforces rating scale.

## 2. Data collection and preprocessing

### 2.1 Standings

The script retrieves standings using the Codeforces API:

`GET https://codeforces.com/api/contest.standings`

Parameters:

| Parameter | Value |
|---|---|
| `contestId` | Target Gym ID |
| `showUnofficial` | `true` |
| `participantTypes` | `CONTESTANT,VIRTUAL,OUT_OF_COMPETITION` |

Gym standings require authenticated API access. The account associated with the API key must have permission to view the contest.

The response contains three main objects:

- `contest`: contest metadata, including name, scoring type, and start time;
- `problems`: problem identifiers and names;
- `rows`: participant standings, scores, penalties, and per-problem results.

The script currently supports only contests whose scoring type is `ICPC`.

### 2.2 Participant classification

Each standings row represents a participant or team.

Participants are classified into three groups:

| Group | Description |
|---|---|
| `rated` | Non-ghost teams whose members all have usable Codeforces ratings |
| `unrated_online` | Non-ghost teams with incomplete rating information |
| `ghost` | Original contestants represented as ghosts |

A team is considered to have solved problem $j$ when its corresponding `problemResults[j].points` is greater than zero.

Define the binary outcome

$$
y_{ij} =
\begin{cases}
1, & \text{if team }i\text{ solved problem }j,\\
0, & \text{otherwise}.
\end{cases}
$$

Rows with no solved problems and no recorded rejected attempts are excluded as likely no-shows.

This is a heuristic: it does not guarantee that every excluded team was inactive.

### 2.3 Observation weights

Different participant groups receive different weights:

| Participant type | Default weight |
|---|---:|
| Online contestant | 1.0 |
| Virtual participant | 1.0 |
| Out-of-competition participant | 0.5 |
| Ghost | 0.5 |

Let $w_i$ represent the observation weight of team $i$.

Weights determine how strongly a team's results influence the estimated problem difficulties. They do not directly multiply or reduce the team's estimated rating.

These are modeling choices rather than official Codeforces parameters.

## 3. Individual rating estimation

The model supports four strategies for obtaining a member's Codeforces rating.

### 3.1 Current rating

The `current` mode retrieves the latest published Codeforces rating using:

`GET https://codeforces.com/api/user.info?handles=...`

The endpoint supports multiple handles in one request. The implementation uses batches of 50.

The returned `rating` field becomes the member's initial rating estimate.

This mode is fast, but a user's current ability may differ from their ability when they participated in an older Gym.

### 3.2 Historical ratings

The historical modes use:

`GET https://codeforces.com/api/user.rating?handle=...`

Each rating-history entry contains:

- `ratingUpdateTimeSeconds`;
- `oldRating`;
- `newRating`.

Let $t$ represent the team's participation timestamp.

Only rating updates satisfying

$$
t_{\text{update}} < t
$$

are considered.

The implementation uses `party.startTimeSeconds` when available.

For official `CONTESTANT` rows without an individual start time, it falls back to `contest.startTimeSeconds`.

For virtual and out-of-competition rows without a reliable start time, it does not assume the original contest's start time.

### 3.3 Historical rating strategies

Let:

- $R_{\text{time}}(t)$ be the latest published rating before participation;
- $R_{\text{peak}}(t)$ be the highest published rating before participation.

The supported strategies are:

**At-time**

$$
R_{\text{effective}}(t)=R_{\text{time}}(t)
$$

**Peak-at-time**

$$
R_{\text{effective}}(t)=R_{\text{peak}}(t)
$$

**Blend**

$$
R_{\text{effective}}(t)
=(1-\alpha)R_{\text{time}}(t)
+\alpha R_{\text{peak}}(t)
$$

The default is $\alpha=0.75$, resulting in:

$$
R_{\text{effective}}(t)
=0.25R_{\text{time}}(t)
+0.75R_{\text{peak}}(t).
$$

This gives more importance to previously demonstrated peak ability while preserving some information about the rating immediately before participation.

The blend is our own heuristic. It is not an official Codeforces formula.

If a participant has no usable rating history before participation, their rating is considered unknown.

## 4. Codeforces team-rating aggregation

For teams with multiple members, individual ratings must be combined into a single team rating.

The implementation follows the method described by Mike Mirzayanov in:

[How Codeforces calculates ratings for teams](https://codeforces.com/blog/entry/16986)

### 4.1 Pairwise win probability

Codeforces uses an Elo-style formula to estimate the probability that a participant with rating $A$ defeats someone with rating $B$:

$$
P(A>B)=\frac{1}{1+10^{(B-A)/400}}.
$$

### 4.2 Team aggregation

Suppose a team has $k$ members with ratings $r_1,r_2,\ldots,r_k$.

The team rating $T$ is defined as the rating of a hypothetical opponent who has a 50% probability of defeating the whole team under the aggregation model.

The opponent defeats the team only if they defeat every member.

Consequently,

$$
\boxed{
\prod_{m=1}^{k}
\frac{1}{1+10^{(r_m-T)/400}}
=\frac12
}
$$

The implementation finds $T$ using binary search.

This construction has several useful properties:

- For a single member, the team rating equals the member's rating.
- A team rating is at least as high as its highest-rated member.
- Adding weaker teammates generally increases the team rating by less than adding similarly skilled teammates.

If any member lacks a usable rating, the current implementation treats the entire team as unrated.

## 5. Problem difficulty model

### 5.1 Rasch probability model

Let:

- $R_i$ be the effective strength of team $i$;
- $D_j$ be the difficulty of problem $j$;
- $S$ be the probability-curve spread.

The probability that team $i$ solves problem $j$ is modeled as:

$$
\boxed{
p_{ij}=\frac{1}{1+10^{(D_j-R_i)/S}}
}
$$

The default spread is $S=500$.

Interpretation:

- If $R_i=D_j$, the predicted solve probability is 50%.
- If $R_i>D_j$, the probability exceeds 50%.
- If $R_i<D_j$, the probability falls below 50%.

The spread controls how sharply solve probability changes with the rating difference.

Increasing $S$ generally increases the distance between estimated easy and hard problem difficulties.

The curve is inspired by Elo and Bradley–Terry models, but its application to ICPC problem-solving probabilities is our modeling assumption.

### 5.2 Likelihood function

The outcome $y_{ij}$ is modeled as a Bernoulli random variable:

$$
y_{ij}\sim\operatorname{Bernoulli}(p_{ij}).
$$

Assuming conditional independence of solve outcomes given team strengths and problem difficulties, the weighted log-likelihood is:

$$
\mathcal L
=\sum_i w_i\sum_j
\left[
y_{ij}\ln p_{ij}
+(1-y_{ij})\ln(1-p_{ij})
\right].
$$

The fitting process attempts to find strengths and difficulties that explain the observed standings while penalizing implausibly large deviations.

## 6. Regularization and unrated participants

Simply maximizing the likelihood can produce unstable estimates, especially when teams solve either all problems or none.

To address this, the model uses Gaussian priors.

### 6.1 Rated teams

For rated team $i$, let $A_i$ be the combined Codeforces team rating calculated from its members.

The model assumes:

$$
R_i\sim\mathcal N(A_i,180^2).
$$

The Codeforces-derived team rating acts as an anchor, but the fitted effective strength can deviate from it according to observed performance.

The standard deviation of 180 controls the strength of this regularization.

### 6.2 Unrated teams and ghosts

For unrated online teams and ghosts, no external rating anchor is available.

Instead, their effective strengths are inferred:

$$
R_i\sim\mathcal N(\mu_g,300^2),
$$

where $\mu_g$ is the mean effective strength of the corresponding group.

The model maintains separate group means for ghosts and unrated online teams.

### 6.3 Group-level regularization

Let $R_{\text{ref}}$ be the median externally calculated rating among rated teams with positive observation weight.

Each unrated group's mean is regularized toward this reference:

$$
\mu_g\sim\mathcal N(R_{\text{ref}},500^2).
$$

This provides a connection between the rated and unrated populations while allowing groups to have different effective strengths.

For example, a ghost population that solves substantially fewer problems than similarly modeled online teams may receive a lower inferred group strength.

This is an *effective strength under the model*, not necessarily the ghosts' true Codeforces-equivalent ability.

### 6.4 Smoothed problem estimates

Problems solved by everyone or nobody create another instability.

For each problem, the model adds one artificial reference participant located at $R_{\text{ref}}$, with a fractional success of 0.5 and failure of 0.5.

This is equivalent to adding:

$$
\frac12\ln p_{\text{ref},j}
+\frac12\ln(1-p_{\text{ref},j})
$$

to the log-likelihood for that problem.

Here,

$$
p_{\text{ref},j}
=\frac{1}{1+10^{(D_j-R_{\text{ref}})/S}}.
$$

This smoothing keeps difficulty estimates finite even for extreme solve rates.

It is a heuristic pseudo-observation, not an independently measured result.

## 7. Optimization algorithm

The implementation fits the model using alternating one-dimensional optimizations.

Rather than applying a general-purpose multidimensional optimizer, it repeatedly updates problem difficulties, team strengths, and group means.

### 7.1 Updating problem difficulties

For each problem $j$, the program finds $D_j$ satisfying:

$$
\sum_i w_i p_{ij}+c\,p_{\text{ref},j}
=
\sum_i w_i y_{ij}+\frac c2,
$$

where $c=1$ is the pseudo-observation weight.

The left side represents expected weighted solves, including smoothing.

The right side represents observed weighted solves plus half a pseudo-solve.

Since predicted solves decrease monotonically as difficulty increases, the equation can be solved using binary search.

### 7.2 Updating team strengths

For each team, let $\mu_i$ be its prior mean and $\sigma_i$ its prior standard deviation.

The updated strength solves:

$$
w_i\sum_j(p_{ij}-y_{ij})
+\frac{S}{\ln 10}\frac{R_i-\mu_i}{\sigma_i^2}
=0.
$$

The first term measures the discrepancy between expected and observed solves.

The second term penalizes deviation from the prior strength.

Again, the function is monotonic and can be solved by binary search.

### 7.3 Updating unrated group means

For an unrated group $g$ with $n_g$ participants, the group mean is updated using:

$$
\mu_g=
\frac{
\sum_{i\in g}R_i+\lambda R_{\text{ref}}
}{
n_g+\lambda
},
$$

where

$$
\lambda=
\left(\frac{300}{500}\right)^2.
$$

This is a shrinkage estimator combining inferred team strengths with the externally anchored reference rating.

The group-mean update uses each team's inferred strength once; observation weights affect the fit through the likelihood instead.

### 7.4 Iteration and convergence

The implementation:

1. Initializes problem difficulties at the rated-team median.
2. Optimizes each problem difficulty.
3. Optimizes each team strength.
4. Updates the means of the unrated populations.
5. Repeats until convergence or the iteration limit.

The convergence criterion is a maximum change below 0.05 rating points for problem difficulties and group means.

The maximum number of iterations is 80. One final difficulty update is performed after the last strength updates.

The numerical searches use fixed bounds and a fixed number of binary-search iterations.

### 7.5 Absolute calibration

A model using only solve outcomes has a fundamental identifiability problem.

If every strength and difficulty increases by the same constant $c$, solve probabilities remain unchanged:

$$
P(R_i+c,D_j+c)=P(R_i,D_j).
$$

Therefore, standings alone cannot establish an absolute rating scale.

**Codeforces-rated online teams provide the external anchors that resolve this ambiguity.**

If no rated online teams are available, the implementation does not generate calibrated problem ratings. It reports solve counts instead.

## 8. Problem-rating output

After fitting, each problem receives a continuous estimated difficulty $D_j$.

For display, the program:

1. Rounds the difficulty to the nearest 100.
2. Clips the displayed value to the range 800–4000.
3. Marks raw estimates outside that range with `*`.
4. Marks problems solved by zero or all usable participants with `?`, indicating high uncertainty.

The printed table contains:

| Column | Meaning |
|---|---|
| `Problem` | Codeforces problem index |
| `Estimated Rating` | Rounded and clipped estimated difficulty |
| `Rated Online` | Solves among online teams with usable rating anchors |
| `Ghosts` | Solves among ghost teams |
| `All Participants` | Solves among all usable standings rows |
| `Name` | Problem name |

Solve counts are accompanied by percentages.

The table's counts are unweighted. Observation weights affect the statistical fit, not the printed solve fractions.

## 9. Team performance estimation

In addition to problem difficulty, the model can analyze a selected team's contest performance.

The output includes:

- Combined team rating or inferred team strength;
- Number of solved problems and penalty;
- Actual placement;
- Expected placement;
- Rank-equivalent performance rating;
- Difference between performance and team rating.

### 9.1 Team rating versus fitted strength

For teams with complete rating information, the displayed team rating is calculated from the members' Codeforces ratings before fitting.

For ghosts and unrated online teams, the displayed value is their strength inferred by the model.

These values have different interpretations: one is externally anchored, while the other is inferred from the Gym's own results.

### 9.2 Actual placement

Actual placement is reconstructed across all usable modeled standings rows.

For ICPC scoring, a team ranks ahead of another if it has:

1. More points; or
2. Equal points and lower penalty.

Let $a_i$ denote the number of opponents strictly ahead of team $i$.

Then:

$$
\operatorname{rank}_i=1+a_i.
$$

If several teams share the same points and penalty, they are considered tied.

For performance calculations, the implementation uses the midpoint of the tied positions:

$$
m_i=\operatorname{rank}_i+\frac{t_i-1}{2},
$$

where $t_i$ is the number of teams tied at that position.

The Codeforces API-reported rank is also printed separately.

Because the modeled population may mix original and virtual participants, this combined placement is a hypothetical comparison, not necessarily an official single-event placement.

### 9.3 Expected placement

Expected placement follows the Elo-style pairwise ranking concept used by Codeforces.

For team $i$ with rating $R_i$, against opponents with strengths $R_j$:

$$
\boxed{
E[\operatorname{rank}_i]
=
1+\sum_{j\ne i}
\frac{1}{1+10^{(R_i-R_j)/400}}
}
$$

Each term estimates the probability that opponent $j$ finishes ahead of team $i$.

The sum is the team's expected placement, often called its *seed*.

The expected-rank calculation always uses the conventional 400-point Elo scale, **independently of the problem-solving spread parameter**, whose default is 500.

For rated opponents, the model uses their externally calculated team ratings.

For ghosts and unrated opponents, it uses their fitted strengths.

All usable opponents count equally in this placement calculation; the ghost and out-of-competition likelihood weights are not applied directly to expected rank.

### 9.4 Performance rating

A team's rank-equivalent performance rating is the hypothetical rating that would produce its observed placement as its expected placement.

For opponents with strengths $R_j$, we seek $R_{\text{perf}}$ satisfying:

$$
1+\sum_{j\ne i}
\frac{1}{1+10^{(R_{\text{perf}}-R_j)/400}}
\approx m_i.
$$

The expected-rank function is monotonic, so $R_{\text{perf}}$ can be found through binary search.

However, the exact first and last positions correspond to infinite limiting ratings in the basic equation.

To keep the calculation finite, the implementation uses a smoothed target rank:

$$
m_i'=1+\frac{(m_i-0.5)(n-1)}{n},
$$

where $n$ is the number of usable modeled teams.

It then finds the rating whose expected placement equals $m_i'$.

The displayed performance difference is:

$$
\Delta_{\text{perf}}
=R_{\text{perf}}-R_i.
$$

A positive value indicates a placement corresponding to a higher rating than the team's estimated prior strength.

**This is not an official Codeforces rating change.**

The method is inspired by Codeforces' expected-rank framework, but the midpoint smoothing, inferred opponent strengths, and treatment of mixed Gym populations are specific to this implementation.

### 9.5 Important interpretation limitation

For unrated teams, the estimated strength depends on their results in the same Gym.

Consequently, their expected placement is not an independent prediction of their actual performance.

Even for externally rated teams, some opponents' strengths are inferred from the same standings, so performance estimates should be considered descriptive rather than independently validated.

## 10. Preview and final calculations

The program performs two stages when a historical rating mode is selected.

### Stage 1: current-rating preview

1. Fetch Gym standings.
2. Retrieve current ratings in batches.
3. Calculate current team ratings.
4. Fit the statistical model.
5. Display preliminary problem difficulties.
6. If requested, display preliminary team performance.

This provides a fast result before historical data collection.

### Stage 2: historical estimate

1. Inspect the rating-history cache.
2. Estimate the historical-fetch duration.
3. Retrieve missing rating histories.
4. Calculate member ratings according to the selected historical mode.
5. Rebuild team rating anchors.
6. Refit the same statistical model.
7. Display final problem difficulties and team performance.

The two stages use the same ghost weights and probability spread.

If `--rating-mode current` is selected, only the first stage runs and its result is final.

## 11. API authentication, caching, and execution time

### 11.1 API authentication

Codeforces Gym standings require an authenticated API request.

The program reads credentials from:

- `CODEFORCES_API_KEY`
- `CODEFORCES_API_SECRET`

Authenticated requests include `apiKey`, `time`, and `apiSig`.

The signature consists of six random characters followed by a SHA-512 digest of the method name, sorted request parameters, and API secret, following Codeforces' documented signing procedure.

The secret must never be included in logs, shared output, or committed files.

### 11.2 API rate limiting

Codeforces documents a limit of approximately one API request every two seconds.

The implementation uses a minimum 2.2-second delay between request starts.

Current ratings are efficient because `user.info` accepts multiple handles.

Historical rating histories require individual `user.rating` calls, so the initial retrieval may be considerably slower.

### 11.3 Historical cache

Rating histories are stored in:

`.cf_rating_history_cache.json`

The cache is located alongside the script unless another path is specified.

It is keyed by case-insensitive Codeforces handles.

The script saves the cache atomically and periodically during retrieval, allowing future runs to reuse previously fetched histories.

The `--refresh-ratings` flag refreshes histories for the requested handles.

Cached histories may become outdated when users participate in subsequent rated contests.

### 11.4 Remaining-time estimate

Let $N_{\text{missing}}$ be the number of uncached rating histories.

The initial completion-time estimate is approximately:

$$
T_{\text{remaining}}
\approx
2.2N_{\text{missing}}+T_{\text{model}},
$$

where $T_{\text{model}}$ is estimated from the duration of the current-rating preview fit.

As requests complete, the program updates the estimate using observed request throughput.

The estimate is approximate and may be affected by network latency, API failures, and model-fitting costs.

## 12. Configuration

| Parameter | Default | Description |
|---|---|---|
| `--rating-mode` | `blend` | Current, at-time, peak-at-time, or blend |
| `--peak-weight` | `0.75` | Historical peak contribution to blended rating |
| `--ghost-weight` | `0.5` | Relative weight of ghost solve observations |
| `--ooc-weight` | `0.5` | Relative weight of out-of-competition observations |
| `--spread` | `500` | Problem-solving probability curve spread |
| `--rating-cache` | Script-local JSON file | Historical rating cache |
| `--refresh-ratings` | Disabled | Refresh cached rating histories |
| `--compare` | Disabled | Refit with ghost weights 0 and 1 for sensitivity analysis |

The default parameters were selected through exploratory analysis of several Gyms, considering difficulty ranges, sensitivity to ghost weighting, and historical rating behavior.

They have not been selected through formal out-of-sample statistical validation.

## 13. Model limitations

The following limitations should be considered when interpreting the results:

**Different participation conditions.** Original contestants and virtual participants may have competed with different levels of prior knowledge, preparation, or pressure. The model cannot directly distinguish these effects from differences in ability.

**Historical rating uncertainty.** Peak ratings can overstate sustained ability, while ratings at participation may underestimate retained competitive programming knowledge.

**Unrated-team inference.** Ghost and unrated-team strengths are estimated from their performance on the same problems whose difficulties are being estimated.

**Conditional independence.** The likelihood treats problem outcomes as conditionally independent. Real contests include dependencies due to shared techniques, strategy, time allocation, and problem ordering.

**Fixed probability curve.** The logistic spread is a heuristic. It has not been established that the chosen curve accurately represents ICPC problem-solving probabilities.

**Extreme solve rates.** Problems solved by zero or nearly all participants have limited direct information. Smoothing prevents infinite values but does not eliminate uncertainty.

**Mixed-team rankings.** Combining ghost and virtual results into a single modeled placement is useful for comparison but does not recreate a single simultaneous contest.

**Incomplete standings.** The implementation assumes that the API response contains the intended standings population. It does not explicitly request additional pages. Large or specially configured contests should be checked for completeness.

**Limited validation.** The model has been tested for numerical behavior and examined empirically on sample Gyms, but its problem ratings and team-performance ratings should not be represented as independently validated equivalents of official Codeforces ratings.

## 14. References

### Codeforces API

- [API Introduction and Authentication](https://codeforces.com/apiHelp)
- [API Methods — contest.standings, user.info, user.rating](https://codeforces.com/apiHelp/methods)
- [API Return Objects — User, RatingChange, Party, RanklistRow, ProblemResult](https://codeforces.com/apiHelp/objects)

### Codeforces rating methodology

- [How Codeforces calculates ratings for teams — Mike Mirzayanov](https://codeforces.com/blog/entry/16986)
- [Open Codeforces Rating System — Mike Mirzayanov](https://codeforces.com/blog/entry/20762)

The Codeforces references describe the official APIs, individual Elo-style comparisons, team aggregation, and expected-ranking concepts.

The regularized Rasch model, historical rating blend, observation weights, smoothing, and Gym performance methodology are components of this estimator rather than official Codeforces procedures.
