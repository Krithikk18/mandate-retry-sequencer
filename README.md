# Mandate Retry Sequencer — synthetic data generator

Ground-truth training and evaluation data for a system that decides **when and
whether** to retry a failed recurring debit, instead of firing a fixed
day-2/4/7 schedule at every failure.

Contains the **data generator**, the **fixed-schedule baseline**, the
**evaluation harness** that scores any policy against the oracle, and the
**sequencer** itself - payday inference, bank-health inference, the money
equation, and the full RETRY / NUDGE / WAIT / HOLD / CANCEL action set, with
ablations for every component. Four of the six failure classes are implemented
behind a router that dispatches on the return code; the other two are declared
stubs and produce no numbers.

---

## The oracle property

This is the reason the data is synthetic, and the single most important thing
about the repository.

A real payments company can only ever observe the retry dates it actually
attempted. If it retried on day 2 and failed, it will never know whether day 5
would have worked. Counterfactual evaluation of a retry policy is therefore
impossible on real data — you can compare policies only by running them live,
slowly, on real customers, at real cost.

Here, we invent each customer's salary date, salary amount, daily burn rate,
cash cushion and shock process. Their bank balance is not a label attached to
the observed attempts; it is a **complete daily series** that exists for every
day in the window, including all the days nobody debited on. So the question

> *would a debit of Rs X on date D have succeeded?*

has an exact answer for **every** D, not just the ones some policy happened to
pick. That makes counterfactual policy evaluation exact rather than estimated:

```python
from generator.generate import generate_dataset
from generator.oracle import first_success_date, success_curve

ds = generate_dataset(n_customers=500, seed=42)
event = ds.failure_events[0]
customer, series = ds.customer(event.customer_id), ds.balance_of(event.customer_id)

# every date the agent could have chosen, with the true answer attached
curve = success_curve(customer, series, customer.subscription_amount)

# the best possible retry date - the policy's ceiling
best = first_success_date(customer, series, customer.subscription_amount,
                          after=event.original_debit_date)
```

Two flavours of truth are exposed:

| function | question it answers | use |
| --- | --- | --- |
| `would_succeed(customer, date, amount, series)` | did *this* customer have the money on *this* date? | scoring a policy's realised decisions |
| `success_probability_true(customer, date, amount, n_sims)` | how often does a customer like this have the money on this date, across fresh draws of their own process? | checking whether the agent's predicted probabilities are **calibrated** |

**Preserving the property.** Two deliberate choices keep the oracle
policy-independent, so the truth cannot shift underneath a policy being
evaluated:

1. The balance process does **not** subtract the Rs 500 NACH bounce charge that
   a failed debit levies on the customer. If it did, the balance path would
   depend on which retries a policy chose, and the oracle would stop being a
   fixed surface. The charge is exposed as `rules.bounce_charge(rail)` for the
   evaluator to apply as a cost on top.
2. Opt-outs are a hazard model (`generator.optout`), not baked into the
   generated history. Cancellation is a consequence of the policy under test,
   so it belongs to the evaluation environment, not to the fixed data.

**Keeping the agent honest.** The truth is only useful if the agent cannot see
it. `agent_view(dataset)` returns an `AgentDataset` whose records are a
*different pydantic type* with no slot for the hidden fields, so a leak is a
construction error rather than a silent data problem:

- hidden customer fields — `salary_day`, `salary_amount`, `salary_delay_prob`,
  `salary_delay_days_max`, `income_shock_start`, `burn_rate`, `buffer_ratio`,
  `optout_sensitivity`, `balance_seed`
- hidden failure field — `balance_at_failure` (a real PSP learns the account was
  short, never by how much)
- and the `balance_series` itself, which the agent view simply does not carry

---

## Install and run

```bash
pip install -e ".[dev]"     # or: pip install numpy pandas pydantic pyarrow pytest
pytest                      # 230 tests
python scripts/inspect_data.py --n 500 --seed 42 --save data/run_42
```

```python
from generator.generate import generate_dataset, save_dataset, load_dataset, agent_view

ds = generate_dataset(n_customers=500, seed=42)   # fully reproducible
print(ds.natural_failure_rate)                     # 0.144
save_dataset(ds, "data/run_42")                    # or fmt="csv"
for_the_agent = agent_view(ds)                     # hidden truth stripped
```

---

## What gets generated

Anchored on `as_of` (default `2026-08-01`), the moment the retry decision is
being made:

```
as_of - 365d ......................... as_of ............ as_of + 60d
|<-------- observed history -------->|  now  |<-- oracle horizon -->|
```

- **`customers`** — 500 subscribers with public fields and hidden generative
  parameters.
- **`payment_history`** — 12 months of attempts, truncated to each customer's
  tenure. Closed cycles were billed on the merchant's legacy fixed schedule
  (original, then +2 / +4 / +7, stopping at the first success), which respects
  the NPCI ceiling of 4 attempts per cycle. The **current** cycle contains only
  its original attempt: what happens next is the agent's job.
- **`failure_events`** — the customers whose current-cycle debit actually
  bounced. Nothing is forced; the rate is whatever the balance process produces.
- **`balance_series`** — the hidden daily truth, running 60 days past `as_of` so
  the oracle can score any retry date the agent picks.

### The numbers, across 20 seeds

Never judge these on one run. `n=500`, seeds 1–20:

| measure | mean | min | max |
| --- | --- | --- | --- |
| natural failure rate (current cycle) | **14.6 %** | 12.2 % | 17.6 % |
| legacy day-2/4/7 recovery | **37.5 %** | 23.9 % | 46.0 % |
| unrecoverable within 30 days | **24.8 %** | 12.7 % | 35.8 % |
| median days from failure to next payday | **10.2** | 8 | 12 |
| failures landing beyond day 7 | **64.2 %** | 55.7 % | 71.9 % |

The failure rate is **not a parameter** — every attempt is resolved against the
customer's simulated balance, and no target appears anywhere in the code. But
the population *was* tuned until the emergent rate landed in the requested
5–20 % band. Both statements are true; state both.

Two of these numbers are quality gates rather than results, and
`tests/test_dataset.py` asserts them:

- **Legacy recovery must sit near 35–45 %.** Published fixed-retry recovery runs
  in that range. A synthetic world where the dumb baseline scores far above it
  has made the problem too easy, and every agent measured against that baseline
  will look better than it deserves. An earlier build of this generator scored
  63.8 % here — the failure-to-payday gap was clustered at ~6 days, so the day-7
  retry landed well by luck. Widening `BURN_RATE_LOG_SIGMA` (0.30 → 0.55) spread
  the dry windows and fixed it.
- **Unrecoverable share must be above zero.** If every failure can be recovered,
  CANCEL is never the correct action and a stopping rule can never be wrong — so
  it can never be tested either. The income-shock population below supplies it.

The failure rate is driven by the *interaction* of two distributions rather than
by noise: salary days cluster at month end (55 % of the book is paid on day 1,
30 or 31), while merchants set debit days uniformly across days 1–28.

### Income shocks: the population that makes stopping testable

About 4 % of customers carry a hidden `income_shock_start`. From that date the
salary stops **and the cushion goes with it** — no income, no floor — so the
balance decays to zero and stays there. For these customers **no retry date can
ever succeed**, which makes CANCEL/HOLD the only correct action and gives a
stopping rule something it can get wrong.

The shock is an **absolute date**, not a window offset. That matters: the oracle
simulates its own 90-day window for `success_probability_true` while
`would_succeed` reads the dataset window, and a window-relative shock would make
the two disagree about whether the customer still had a job. There's a test for
exactly that.

The agent never sees the flag. Its only evidence is the signature the shock
leaves in `payment_history`: a run of consecutive cycles where all four attempts
fail and none recover. A test asserts that trail actually exists for at least
half of shocked customers — a hidden state the agent cannot possibly infer would
be unfair rather than hard.

---

## The baseline and the scoring harness

Every policy - the fixed schedule, the oracle ceiling and the agent built next -
implements one interface: `decide(PolicyState) -> RetryDecision`. `PolicyState`
is built only from `agent_view` output and holds `Agent*` records, so a policy
cannot reach the truth; pydantic also refuses to coerce a truth-family
`Customer` into an `AgentCustomer` slot, so it cannot be smuggled in either.

### The scale every agent is scored on

`python scripts/run_baseline.py` - 20 seeds x 500 customers:

| | mean | range |
| --- | --- | --- |
| fixed-schedule floor (day-2/4/7) | **35.0 %** | 22.4 - 42.9 % |
| oracle ceiling | **73.6 %** | 62.7 - 85.7 % |
| **headroom to capture** | **38.6 pp** | |

Supporting numbers, same run:

| metric | legacy | oracle |
| --- | --- | --- |
| presentments used | 253 | 126 |
| wasted attempts (true P < 0.20) | 126 | 0.8 |
| wrongly cancelled | 24.2 | 0 |
| customers retained | 25.5 | 71.6 |
| cost per rupee recovered | Rs 0.57 | Rs 0.07 |
| net value | -Rs 1.76 m | +Rs 0.12 m |

The legacy baseline is **not** crippled. It obeys the same NPCI attempt ceiling,
the same 24-hour notice lead time and the same execution window, and the
+2/+4/+7 offsets are the published industry convention rather than a strawman.
It burns half its presentments on days when the true probability of clearing is
under 20 % because a schedule fixed in *retry index* cannot track an event that
moves in *calendar time*.

**The ceiling is verified, not assumed.** Oracle recovery plus opt-out share
equals the dataset's recoverable share to within 1e-9 on every seed - the
simulator loses nothing. A test pins that identity, so if a future change starts
dropping recoveries it fails immediately rather than quietly lowering the bar.

### Compliance violations raise

A policy that schedules a fifth presentment, or one inside the 24-hour notice
window, or outside 00:00-07:00, gets a `ComplianceViolation`. Nothing is
silently clamped. Clamping would convert an illegal policy into a legal-looking
one with a good number attached, which is the worst thing an evaluation harness
can do.

### Opt-out draws are common random numbers

The draw is keyed on `(customer_id, notification_index, seed)`, never on a
shared stream. Off a shared generator, a policy making three attempts would
shift the stream for the next policy and the comparison would measure RNG
alignment rather than skill. Two policies that put their Nth notice on the same
customer draw the same uniform, so any difference in outcome is a difference in
the probability they created.

### One trap worth knowing about

Customer ids repeat across seeds - `CUST_0001` is a **different person** in every
dataset. An early version of `run_baseline.py` merged 20 truth tables into one
dict keyed by customer id, so the oracle read the wrong person's balance on 19 of
20 seeds and scored *below* the baseline. `OraclePolicy.for_dataset(dataset)`
binds per seed, `compare()` accepts a factory so it can rebind, and
`OraclePolicy.decide` now raises if the truth it holds disagrees with the state
it was handed. Fail loudly, not quietly.

### HOLD versus CANCEL

    CANCEL -> mandate dead,       ltv_lost = annual_ltv
    HOLD   -> nothing this cycle, mandate survives, ltv_lost = 0

For a customer whose income has genuinely stopped, no retry date succeeds. The
fixed schedule burns three presentments and then kills a mandate that was
structurally alive - the customer never chose to leave. That asymmetry is most
of the gap between the floor and the ceiling, and it only shows up because the
accounting refuses to treat the two actions as equivalent.

Two accounting conventions, both deliberate: `bounce_charges_paid` is a charge
on the *customer* but is subtracted from *merchant* net value, because a
sequencer that racks up Rs 500 bank charges on the people it is retaining is
destroying what it is paid to protect; and `attempts_used` includes the original
failed presentment, which is a constant offset across policies and so never
moves a comparison.

### Scripts

```bash
python scripts/run_baseline.py --seeds 20 --n 500
python scripts/run_baseline.py --seeds 5 --irritation build      # the ablation
python scripts/inspect_episode.py --customer CUST_0013 --policy legacy
python scripts/inspect_episode.py --policy legacy --worst
```

`inspect_episode.py` is the audit trail: every decision, the reasoning behind
it, the hidden balance on that day, the true success probability of each
scheduled date, and the outcome. On CUST_0013 the baseline presents on +2 and +4
at a true probability of 0.00 each before catching payday on +7; the oracle
presents once, on payday, with one notification instead of three.

---

## Holding has a price

An earlier build made HOLD free, which meant it strictly dominated CANCEL and
no stopping rule could ever be wrong - so none could be tested. HOLD is now
what CONTEXT says it is: not a terminal action, but a decision to carry the
mandate into the next cycle. Episodes run up to `MAX_EPISODE_CYCLES` (12,
matching the horizon `annual_ltv` is denominated in), and carrying costs
compound three ways per cycle:

1. `HOLD_CARRYING_COST_FRACTION` (0.15) of the subscription price - marginal
   cost of serving somebody who is not paying.
2. A fresh presentment, because a live mandate keeps auto-billing on its anchor
   date. On NACH each one bounces and levies Rs 500 on the customer. **This
   term dwarfs the other two on that rail.**
3. `NATURAL_ATTRITION_PER_CYCLE` (5%) - held customers drift away on their own,
   and the annual LTV is lost anyway, just later and after paying to get there.

### The finding: 0.15 is not enough to make cancelling correct

`python scripts/sensitivity_holding.py` sweeps the estimate. Two results:

**The recovery-rate gap does not move at all** across 0.05-0.30. The carrying
cost changes which action is optimal, not how much money is on the table. So
the floor/ceiling recovery numbers can be quoted without caveat; the net-value
numbers move with it and cannot.

**On income-shocked customers - the ones who can never be recovered - holding
still beats cancelling at every value in the specified sweep:**

| carrying cost | hold forever | cancel after 2 cycles | better |
| --- | --- | --- | --- |
| 0.15 (default) | -Rs 22,767 | -Rs 31,545 | holding |
| 0.30 | -Rs 26,239 | -Rs 31,941 | holding |
| 0.45 | -Rs 29,711 | -Rs 32,337 | holding |
| **0.60** | -Rs 33,183 | -Rs 32,732 | **stopping** |
| 1.00 | -Rs 42,442 | -Rs 33,788 | stopping |

The crossover sits near **0.60 of the subscription per cycle** in aggregate -
near 0.45 on NACH, where the bounce charge compounds, and near 0.75 on UPI. The
arithmetic is not subtle: CANCEL forfeits 12x the subscription immediately,
while a full year of carrying costs 1.8x plus presentment fees.

So the stopping rule is now *capable* of being right, which it was not before,
but at the specified parameters it is still rarely the profitable move. That is
a property of the cost model, not a bug. Three tests pin it: one that holding
loses at a high enough cost, one that it still wins at 0.15, and one that the
penalty is monotone in the fraction.

---

## The payday model (agent layer 1)

`src/agent/payday.py`. It decides nothing - it estimates, per candidate date,
the probability a debit of a given size would clear. The decision engine
multiplies that by real rupees, so **calibration matters more than
confidence**.

Everything is inferred from attempt dates and outcomes. The only outside
knowledge is a documented prior over Indian payroll dates, hardcoded rather
than learned, because twelve sparse cycles cannot identify a payday alone.

### Three tiers, and what the calibration says

`python scripts/calibrate_payday.py` - 3 seeds x 300 customers, 30-day horizon
per failure, fitted only on history up to the failure date:

| tier | what it does | Brier | ECE | skill vs base rate |
| --- | --- | --- | --- | --- |
| 1 | day-of-month frequency, Beta-smoothed | 0.2852 | 0.2149 | **-15.1%** |
| 2 | **posterior over which day is payday** | 0.2221 | **0.1131** | **+10.4%** |
| 3 | tier 2 + fitted liquidity depth | **0.2187** | 0.1693 | +11.7% |

**Tier 1 is worse than useless** - negative skill means it scores worse than
ignoring the date entirely and predicting the base rate. It scores each day
independently, so it cannot generalise from the four days the legacy +2/+4/+7
ladder happened to probe, and it is confidently wrong about the other 26.

**Tier 2 wins on both Brier and ECE.** Inferring *which* day is payday and
deriving liquidity from the phase since it is worth more than scoring days
independently. That is the thesis, and it holds on held-out seeds (4-6 and 7-9
give skill +13.5% and +10.6%, with tier 2 best on ECE in every group).

**Tier 3 discriminates slightly better but is less calibrated.** It wins Brier
on some seed groups and loses on others, and its ECE is worse than tier 2
everywhere - it is systematically underconfident. Since the money equation
needs a well-scaled probability rather than a good ranking, **tier 2 is the one
to wire into the decision engine.**

The liquidity shape (`LIQUIDITY_MIN`, `DEFAULT_DEPTH_DAYS`) was swept rather
than assumed; the current values are already the Brier+ECE optimum for tier 2.

### How tier 2 works

Prior over payday: the Indian payroll distribution, plus a floor over all 31
days so an unusual payday stays findable - without that floor a customer paid
on the 18th could never be located however clear the evidence, which is the
classic way a strong prior becomes a bug.

Likelihood: for a candidate payday `d`, an attempt on date `t` sits `phase`
days after the most recent `d`. A success is likely when phase is small, a
failure when the month has been spent. Multiply across attempts, weight by the
prior, normalise. The liquidity curve then **marginalises over the posterior**
rather than committing to a single guess - a model that picks one payday and is
wrong is confidently wrong, while one that spreads over its posterior is merely
uncertain, and uncertainty is what the money equation can price.

The identifying case is exactly the one CONTEXT describes: a customer paid on
the 7th and debited on the 2nd. The original bounces, the ladder walks forward
across payday, one retry clears. Successes and failures at different phases are
what locate the salary date - a history of nothing but failures cannot, and the
model correctly stays near the prior when that is all it has.

### Income shock, inferred

`shock_probability()` reads the only observable signature: consecutive cycles
that were *properly tried* (two or more attempts) and failed outright. A Bayes
factor over that count. The current cycle's single bounce is deliberately not
evidence - it is why we are here. When the probability is high the liquidity
curve collapses toward zero everywhere, which is the correct marginalisation.

It never reads `income_shock_start`. A test asserts nothing under `src/agent/`
imports the truth at all.

### One caveat to state in any write-up

The synthetic population is drawn from the same payroll shape the model uses as
its prior, so the agent starts with the correct prior *family*. On real data
the prior would be estimated from payroll statistics and would be somewhat
wrong, costing tier 2 some of its advantage. The tier 1 vs tier 2 comparison is
still fair - both see the same data - but the absolute calibration numbers are
optimistic.

---

## The sequencer

`src/agent/`. Three layers behind one `decide(PolicyState) -> RetryDecision`:

1. **`payday.py`** turns attempt history into a liquidity curve.
2. **`decision_engine.py`** prices every legal date in rupees.
3. **`sequencer.py`** picks an action and writes down why.

### The money equation

For each candidate date `d`:

    value(d) = P_clear(d) x amount                     expected recovery
             - processing_cost[rail]                   cost of presenting
             - (1 - P_clear(d)) x bounce_charge[rail]  expected bounce charge
             - P_optout(...) x annual_ltv              expected churn

`P_clear` comes from the tier-2 payday model. `P_optout` comes from
**`agent/churn.py`**, not from `generator.optout` - the true hazard reads a
hidden per-customer `optout_sensitivity`, so the agent substitutes the
book-level average and sees only tenure and notification count. A test forbids
`src/agent/` from importing `generator.optout` at all.

Two things the equation sees that a fixed schedule cannot:

- **The bounce term is rail-dependent.** On NACH a date needs roughly a 70%
  chance of clearing before it is worth presenting on, because the 30% failure
  branch puts Rs 500 on the customer's bank statement. The same date is worth
  taking on UPI and not on NACH.
- **The churn term scales with annual LTV, not the amount at stake.** Chasing
  Rs 199 through four notifications risks Rs 2,388 of subscription value.

The engine returns the **whole scored list**, not the argmax. "We chose the 7th"
is an assertion; "we chose the 7th at +Rs 210 over the 4th at -Rs 38" is a
reason. The top five land in `RetryDecision.reasoning` along with P_clear,
P_optout, the payday posterior mode and its confidence, the shock probability,
the attempt and notification budget, and which constraint bound the choice.

### Results, 20 seeds x 500 customers

Measured on the full mixed population - 81.7% insufficient funds, 9.3% bank
downtime, 6.5% expired card, 2.5% revoked consent:

| | recovery | money | net value | attempts | optouts | retained |
| --- | --- | --- | --- | --- | --- | --- |
| legacy fixed | 35.5% | 78,940 | -1,684,613 | 272 | 7.1 | 26.5 |
| **sequencer** | **58.9%** | **155,127** | **-530,506** | 336 | 2.1 | 59.9 |
| oracle ceiling | 67.8% | 175,003 | -337,214 | 285 | 1.5 | 65.6 |

**Headroom captured: 72.5%.** Comfortably between the floor and the ceiling,
which is where it has to be - at or above the oracle would mean information is
leaking, and a test asserts it never happens on any seed.

The ceiling sits lower here (67.8%) than on an insufficient-funds-only book,
because about 9% of failures are structurally dead mandates that **nothing** can
recover. Not even the oracle. That is the correct ceiling for a mixed book, and
quoting the higher one would be quoting an easier problem.

Beyond recovery: **3x fewer opt-outs induced** (2.1 vs 7.1) and **2.3x more
mandates retained** (59.9 vs 26.5). On an insufficient-funds-only book the
sequencer also made 3.6x fewer *wasted* attempts - presentments where the true
probability of clearing was under 20% - though that metric needs the slower
oracle-metrics pass and is not in the table above.

### Ablations: what each piece is worth

`python scripts/run_ablations.py`. Deltas are **paired within each seed**, so
seed noise cancels; `seeds worse` counts how many of the 20 the variant lost on,
because a real effect shows up on most seeds rather than only in the mean.

| variant | d recovery | d net value | seeds worse | verdict |
| --- | --- | --- | --- | --- |
| `no_payday` | -17.1% | -464,675 | 19/20 | the biggest single contributor |
| `no_hold` | -8.9% | -527,251 | 20/20 | HOLD earns its place |
| `tier1_payday` | -7.4% | -154,745 | 17/20 | tier 2 over tier 1 is worth real money |
| `with_nudge` | -3.0% | -77,140 | 17/20 | **NUDGE is a liability** |
| `irritation_build` | -1.3% | -41,208 | 16/20 | borderline - see below |
| `no_optout_cost` | +0.3% | -19,543 | 16/20 | borderline - see below |
| `no_bank_health` | +0.0% | **0** | 0/20 | **no effect at all - see class 2** |

Four findings worth stating plainly:

**NUDGE hurts, so it is off by default.** It costs 4.1 points of recovery on 19
of 20 seeds: nudging spends a day without presenting, and the modelled lift does
not pay for the delay. The switch is kept so the finding stays reproducible
rather than deleted.

**The opt-out cost term is marginal.** It costs essentially nothing in recovery
and returns Rs 35,819 of net value, but only on 15 of 20 seeds - exactly the
threshold this script uses to call an effect consistent. It halves induced
opt-outs (1.6 vs 3.4), which is the mechanism working as intended, but the LTV
saved barely outruns the recovery forgone. **N2 should be claimed as a modest,
measured contribution, not a centrepiece**, and the honest sentence is "it pays
for itself narrowly" rather than "it is worth Rs 35,819".

**The irritation direction barely matters, and the verdict is borderline.**
16 of 20 seeds, against a 15-seed threshold. Read it as "the central conclusion
does not obviously depend on which way the assumption points", not as a clean
robustness result - the two directions are close enough that a different seed
range could tip the label either way. That is itself the useful finding: it is
not a load-bearing assumption.

**The shared bank-health signal changes nothing - identically, on every seed.**
Not "small", not "borderline": zero difference to the rupee on all 20. The
inference itself works and is tested; it simply has nothing to act on, because
the notice period is longer than any outage. That is a structural result about
RBI Sec 6(a) rather than a defect in the model, and it is worked through in the
class-2 section below.

### Novelty N3: what the 24-hour notice actually costs

`python scripts/lead_time_ablation.py`. Two separable costs, and the interesting
result is which one is expensive.

**The 24-hour timing requirement costs approximately zero.** Notice periods of
0, 6, 12, 24 and 48 hours give identical recovery. The reason matters more than
the number: what binds a retry sequencer is knowing *which day* the salary
lands, not how far ahead it must commit. This agent has no bank-side signal, so
lifting the notice period buys it scheduling freedom it has no new information
to use. **Read it as a floor** - an agent with a real-time balance feed could
react to a credit the moment it landed, and for that agent the rule would bind
hard. Measuring that needs a signal this generator does not model.

**The mandated opt-out costs Rs 6,317 per 100 failures** (1.35 points of
recovery), measured against a counterfactual world where the notice carries no
way out. That is Sec 6(c), not Sec 6(a).

For a retry sequencer, **the expensive half of the e-mandate framework is the
opt-out, not the timing.** As far as we can find, that number is not published
anywhere.

### Prior robustness: is it the method or the prior?

The population is drawn from the same payroll distribution the model uses as its
prior, which flatters the agent. `python scripts/prior_robustness.py` re-runs
with deliberately wrong ones.

| prior | recovery | vs correct | mean payday error | exact |
| --- | --- | --- | --- | --- |
| correct | 65.8% | | 2.84 days | 43.7% |
| flat (uniform 1-28) | 65.6% | -0.2% | 5.01 days | 24.6% |
| shifted (+5 days) | 64.8% | -1.0% | 4.49 days | 11.3% |
| us_style (1st/15th) | 66.1% | +0.3% | 4.32 days | 37.3% |

**97% of the advantage survives the worst wrong prior.** And the reason is not
that the prior is ignored - it clearly is not, since a wrong one roughly halves
how often the exact payday is identified (43.7% -> 11.3%). It is that **the
liquidity window is about a fortnight wide, so being two or three days out still
puts the presentment on a day with money in the account.** The decision is
coarser than the inference, and that is what makes it robust.

So the method is doing the work. The payroll prior is a head start, not the
result - which is the honest answer to "how much of this transfers to a market
whose payroll calendar you do not know".

---

## The audit trail

`python scripts/show_audit.py --customer CUST_0013 --seed 42`

Every number in the rendered trail is **read from what the policy recorded at
decision time**, never recomputed. A trail that re-derives its own figures can
quietly disagree with the run it claims to describe, which makes it worse than
no trail at all: it looks like evidence while being a second opinion.

```
══════════════════════════════════════════════════════════════════════
CUST_0013 · Rs 339 · upi_autopay · 5 months · LTV Rs 4,068
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

02 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

02 Jul   DECISION: WAIT  (unchanged for 7 days)
         Payday posterior: day 7 (conf 0.73)
         Shock probability: 0.05
         Best candidate 10 Jul scored +Rs 191
           P(clear) 0.88 x Rs 339            =    +Rs 298
           processing                        =      -Rs 5
           bounce risk 0.12 x Rs 0           =       Rs 0
           opt-out risk 0.025 x Rs 4,068     =    -Rs 102
         Runners-up: 11 Jul +Rs 191 · 09 Jul +Rs 190 · 12 Jul +Rs 190
         best date is not yet due for commitment

09 Jul   DECISION: RETRY
         ...
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓

10 Jul   PRESENTMENT - SUCCEEDED, Rs 339 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 329 · 8 days to resolution
Mandate retained · LTV preserved Rs 4,068
legacy_fixed on the same customer: presented 04 Jul, 06 Jul, 09 Jul;
recovered, 3 notifications sent, net +Rs 319
══════════════════════════════════════════════════════════════════════
```

Three things it does deliberately: shows the **arithmetic** rather than the
total, so a reader can find the term they disagree with; shows the
**runners-up**, because "we chose the 7th" is an assertion while "we chose the
7th at +Rs 191 over the 9th at +Rs 190" is a reason; and prints the **legacy
comparison** on the same customer, so no episode is shown in isolation.

Consecutive identical decisions collapse into one line with a day count -
summarised, never hidden. `style="json"` and `style="markdown"` render the same
content for machines and for docs.

**[docs/audit_samples.md](docs/audit_samples.md)** holds ten episodes chosen by
CATEGORY, not by outcome. Two of them are deliberately unflattering: one where
the fixed schedule beat the sequencer outright, and one where the sequencer
cancelled a mandate the oracle says was still recoverable. A sample where the
agent only wins is marketing.

---

## The other failure classes

CONTEXT lists six. This build implements three and declares the rest.

| class | modelled? | success on any date | actions |
| --- | --- | --- | --- |
| `insufficient_funds` | **yes** - the payday model | depends on timing | all five |
| `bank_or_psp_downtime` | **yes** - bank health x liquidity | depends on the hour | RETRY, WAIT, HOLD, CANCEL |
| `expired_or_reissued_card` | **yes** (trivially) | **0.0** | NUDGE, HOLD, CANCEL |
| `customer_opted_out` | **yes** (trivially) | **0.0** | CANCEL only |
| `afa_incomplete` | stub | - | declared, not implemented |
| `ambiguous_decline` | stub | - | declared, not implemented |

The two complete-but-trivial classes are the point of the abstraction. An
expired credential and a revoked mandate cannot clear on **any** date, so their
success model is the constant zero and their action set simply omits RETRY.
That is five lines each, and it buys the result below.

The stubs exist so the router has no missing branch. They are marked
`implemented = False`, they produce no numbers, and nothing is claimed for them.

### The router

`src/router.py` dispatches on the **return code**, exactly as a production
system would with an acquirer response - not on a label the generator helpfully
attached. An unmapped or missing code routes to the ambiguous handler rather
than raising, because real acquirers emit codes nobody documented.

It asserts its own consistency at import: every declared class has a handler,
every handler answers to its own name, and **nothing in
`NEVER_RETRY_CLASSES` offers RETRY**. Adding a class cannot silently make a dead
mandate retryable.

### Waste is not the same as a breach

Retrying an expired card is **futile but lawful**. Retrying a mandate whose
consent was withdrawn under RBI Sec 6(c) is a **regulatory breach** - there is
no mandate left to present against. The simulator raises `ComplianceViolation`
on the second and not the first, and `strict_compliance=False` switches to
counting instead of aborting so the breaches can be reported.

### Multi-class results, 20 seeds x 500 customers

`python scripts/run_multiclass.py`. Realised mix: 81.7% insufficient funds,
9.3% downtime, 6.5% expired card, 2.5% revoked.

| | legacy | sequencer |
| --- | --- | --- |
| recovery (all classes) | 35.6% | **59.0%** |
| presentments | 5,442 | 6,711 |
| notifications | 3,947 | **1,496** |
| opt-outs induced | 141 | **42** |
| **compliance breaches** | **41** | **0** |

**On the structurally dead mandates**, routing on the return code saves **300
presentments and Rs 38,730** across 1,636 failures - **Rs 2,367 per 100
failures** - for about fifty lines of code and no model at all. The sequencer's
presentments there are the originals that produced the failure in the first
place; it makes none of its own.

**And 47 compliance breaches become 0.** A fixed schedule cannot avoid this: it
never reads the return code, so it cannot know consent is gone. That is a
regulatory finding rather than an efficiency one, which is why it is reported
separately.

One modelling choice worth naming: a mandate held on a dead class is
**suspended**, not re-presented. Nobody knowingly auto-bills an expired
credential every month, and without that rule holding such a mandate would look
more expensive than cancelling it purely because of presentments a real merchant
would never make.

---

## Failure class 2: bank and PSP downtime

The mirror image of insufficient funds, and built to make that contrast
visible:

    insufficient funds -> recovery takes DAYS.  Wait for payday. Patience wins.
    downtime           -> recovery takes HOURS. Present at the first clear
                          hour. Patience loses.

Same engine, opposite policy - and the shape is derived rather than coded. Both
come from the same money equation being handed a different `P(clear)` curve. No
branch anywhere says "wait during an outage".

### The generator

`src/generator/banks.py`. Outages are a bank-level renewal process at **hour
resolution**, because an outage from 01:00 to 04:00 either does or does not
cover the 02:00 presentment and nothing coarser can express that.

Outage rate scales inversely with book share:

    rate = BASE_RATE * (REFERENCE_SHARE / share) ** ELASTICITY

One knob, so large issuers get better uptime through a mechanism rather than a
hardcoded two-tier table. Measured: SBI ~1.1% downtime, the smallest small
finance banks ~5%. Durations are lognormal, median 3h, tail clipped at 18h.

The outage book is a **pure function of (seed, window)**, never persisted. Any
caller rebuilds it identically from `dataset.seed` and `dataset.meta`, so it
survives `save_dataset`/`load_dataset` for free - `outage_book_for(dataset)`.

`bank_id` is a **public** field on `Customer` and `AgentCustomer`. A real PSP
reads it off the IFSC or UPI handle, and the whole class depends on being able
to correlate failures across customers who share an issuer.

**Realised share: 9.1% of failures carry code 91** (mean over 20 seeds).
Individual seeds range from **1.3% to 22.3%**, and only 3 of 20 land inside the
8-12% target band. That spread is not a tuning failure - it is the correlation
showing up. Insufficient funds is an independent per-customer draw, so its share
is stable; one outage fails every mandate on that issuer at once, so whether a
large bank happened to be down on a busy debit day swings the whole seed.

### The first cross-customer signal

`src/agent/bank_health.py` is architecturally different from everything before
it. `agent/payday.py` reasons about ONE customer in isolation - another
subscriber's salary date says nothing about yours. Here it is the opposite: five
downtime-coded failures on one bank in one hour are evidence about every OTHER
mandate on that bank, including the ones that have not failed yet.

The discriminating property, and the reason the class is separable at all:

| evidence | inference |
| --- | --- |
| cluster of **91s** on one bank | that bank is down |
| cluster of **04s** on one bank | those customers are broke |

Both cluster in time. Only the first says anything about the bank, and that
falls straight out of counting nothing but downtime-coded evidence. Measured:
five 91s in an hour take the outage probability from 1.5% to 92%; twelve 04s on
the same bank in the same hour leave it at exactly the 1.5% prior.

Cold start returns the book-wide prior - never 0, because a quiet bank is not a
healthy one, and never 1, because no finite pile of declines makes an outage
certain.

### Results, 20 seeds x 500 customers, downtime failures only

| policy | recovery | presentments | into a live outage | notices |
| --- | --- | --- | --- | --- |
| legacy fixed | 89.5% | 334 | 1 | 186 |
| **sequencer** | **94.1%** | 307 | 1 | 136 |
| oracle ceiling | 94.7% | 299 | 0 | 149 |

The sequencer beats legacy by 4.6 points and sits within half a point of the
ceiling. But the interesting results are the negative ones.

### Three negative results worth more than the win

**1. No policy can present into the outage it is reacting to. 0 of 19,535
outages cover two presentment windows.**

Outages cap at 18h. The RBI notice period is 24h. Presentments happen once a
day. So an outage that blocks today's debit is *always* over before the earliest
hour a compliant agent could present again. This is structural, provable from
two constants, and it explains why every policy scores within a few points of
every other.

**2. The shared bank-health signal changes nothing, and that follows from (1).**

`no_bank_health` produces results identical to the rupee. The inference works -
tests pin that it fires on 91-clusters and stays flat on 04-clusters - but it
has nothing to act on. By the earliest legal presentment hour the outage is
provably over, so the bank term multiplies every candidate date by the same
near-one number and cannot reorder them.

The signal is real at scale (the largest same-bank same-day cluster of 91s in a
500-customer book is 10; at 5,000 customers it is 45). It would bind if outages
could outlast the notice period - real core-banking incidents sometimes run for
days - but none in this generator do, by construction.

**This is the sharper regulatory finding.** For class 1 the 24-hour rule costs
about nothing because the agent has no bank-side signal to react to. Here it
costs nothing for a structural reason instead: the rule does not make the class
unwinnable, it makes it nearly automatic - and it makes real-time bank health
monitoring worthless for scheduling compliant retries, which is the first thing
a payments company would instinctively build for this failure mode.

**3. The money equation has no time preference, and this class exposes it.**

The agent recovers as often as the oracle but takes **538 hours** after the
outage clears against the oracle's **34**. It is not a downtime bug. Rs 2,571
collected in thirty days scores higher than Rs 2,551 collected tomorrow, because
the churn term decays with time and nothing penalises delay. For insufficient
funds that barely shows - you must wait for payday anyway. For downtime the
money was collectable the next morning.

`DecisionEngine.daily_discount` prices it and **defaults to 0.0**, so no
reported number depends on it. Measured on this class:

| discount/day | recovery | mean days after clear |
| --- | --- | --- |
| 0.00 (as specified) | 96.1% | 23.2 |
| 0.01 | 96.1% | **10.7** |
| 0.03 | 96.1% | 7.5 |

A 1%/day preference roughly halves the delay at no cost in recovery. It is left
off because turning it on would change every class-1 number already reported -
that is a call for the write-up, not a silent edit - but it is the
highest-value change on the list.

### A bug this class surfaced in class 1

`PaydayModel` was treating **every** failed attempt as evidence the account was
empty. A 91 never reached the account and a 54 has no live credential; neither
says anything about liquidity. Counting them dragged the inferred payday around
and made the model wait for a salary that was never the problem. Censored
observations are now dropped rather than counted as balance failures - which
affects class 1 too, not only downtime.

---

## Package layout

```
src/generator/
  rules.py        RBI / NPCI / NACH constants. Regulations, not tunables.
  models.py       Pydantic records + the agent/truth type split.
  balance.py      The hidden daily balance process (vectorised over paths).
  banks.py        Bank outages: hour-resolution, shared across customers.
  oracle.py       Ground truth: would_succeed, success_probability_true, curves.
  optout.py       The cancellation hazard attached to each pre-debit notice.
  population.py   Indian subscriber population. Every distribution is a knob.
  generate.py     generate_dataset / save / load / agent_view.

src/policy/       AGENT-SAFE. Nothing here can import the truth.
  base.py           Policy / PolicyState / RetryDecision. The contract.
  legacy.py         LegacyFixedSchedule: the +2/+4/+7 industry baseline.
  oracle_policy.py  THE CEILING. Cheats. Not exported from the package.

src/router.py     Dispatches a failure to its class handler, on the return
                  code. Asserts its own consistency at import.

src/classes/      One module per failure class, same two-function interface.
  base.py           FailureClassHandler / NeverRetryHandler.
  insufficient_funds.py  payday timing; recovery takes days
  downtime.py       bank availability x liquidity; recovery takes hours
  expired_card.py   dead credential: zero on every date, no RETRY
  opted_out.py      consent revoked: CANCEL only, RETRY is a breach
  stubs.py          AFA / ambiguous - DECLARED, NOT IMPLEMENTED

src/audit/        Renders decision logs for human review.
  trail.py          reads what the policy recorded; never recomputes it

src/agent/        AGENT-SAFE. The intelligent policy.
  payday.py         PaydayModel: when is this customer liquid, and how sure
                    are we? Three tiers; tier 2 is the calibrated one.
  bank_health.py    The first CROSS-CUSTOMER signal: pools decline evidence
                    across everyone sharing an issuer.
  churn.py          Opt-out risk from public fields only. The true hazard
                    reads a hidden sensitivity, so the agent estimates it.
  decision_engine.py  The money equation. Prices every legal date; returns
                    the whole ranked list, not just the winner.
  sequencer.py      SequencerPolicy: picks RETRY/NUDGE/WAIT/HOLD/CANCEL and
                    writes the audit trail. Every ablation is a config flag.

src/evaluation/   May read the truth. Policies may not.
  simulator.py    One policy, one failure event, day by day. Raises on
                  compliance violations rather than clamping.
  economics.py    The money equation, and the HOLD/CANCEL asymmetry.
  harness.py      evaluate() / compare(). Mean and range across seeds.

scripts/inspect_data.py         population, histograms, oracle headroom
scripts/run_baseline.py         floor vs ceiling across 20 seeds
scripts/inspect_episode.py      one customer, day by day, with reasoning
scripts/sensitivity_holding.py  does the carrying-cost estimate change anything
scripts/calibrate_payday.py     tier-by-tier reliability of the payday model
scripts/run_ablations.py        what each component of the agent is worth
scripts/lead_time_ablation.py   the rupee cost of the RBI notice rule (N3)
scripts/prior_robustness.py     method or prior? re-run with wrong priors
scripts/run_multiclass.py       legacy vs sequencer across all failure classes
scripts/run_downtime.py         failure class 2, and what the notice rule costs it
scripts/show_audit.py           the full decision trail for one customer
scripts/export_audit_sample.py  ten episodes to docs/audit_samples.md

tests/test_generator.py   unit-level: balance, oracle, population, opt-out
tests/test_dataset.py     dataset-level: reproducibility, realism, leakage
tests/test_policy.py      the contract and the baseline schedule
tests/test_evaluation.py  compliance, determinism, economics, harness
tests/test_payday.py      payday inference, shock detection, calibration
tests/test_decision_engine.py  the money equation and the churn estimator
tests/test_sequencer.py   the policy, its audit trail, and the guardrails
tests/test_classes.py     the router, the never-retry classes, the audit trail
tests/test_downtime.py    outages, bank health inference, the two-condition oracle

docs/audit_samples.md     ten rendered episodes, picked by category
```

### The balance process

Per customer, per day: salary lands on `salary_day` (clamped to month end in
short months, and late with probability `salary_delay_prob`); spend of
`salary * burn_rate * LogNormal(mean 1.0, sigma 0.3)` comes off; a 5 % chance
of a shock worth 10–30 % of salary; and a floor at
`salary * buffer_ratio * U(0.1, 1.0)` — customers with `buffer_ratio` near zero
genuinely run the account to empty. A 60-day burn-in is simulated and
discarded so the window starts from a settled balance, and surplus above 2.5x
salary is swept to savings when the next salary lands so low-burn customers do
not accumulate without bound. Customers carrying `income_shock_start` lose both
the salary and the floor from that date onward.

The **spread** of `burn_rate` matters more than its median: it sets how long
each customer's dry window is, and therefore the distribution of "days from
failure to next payday" — which is what decides whether a fixed schedule can
reach the money at all. That one parameter is the difference between a dataset
where the dumb baseline scores 64 % and one where it scores 38 %.

The whole thing is vectorised over paths, so 1,000 counterfactual draws of a
425-day series cost about 50 ms — that is what makes
`success_probability_true` cheap enough to use as a calibration target.

### Regulatory rules encoded

All from `rules.py`. Read from there; never hardcode a threshold at a call site.

- **RBI e-mandate 2026** (circular RBI/DPSS/2026-27/396, 21 April 2026) — Sec
  6(a) pre-debit notification at least 24 h before *every* debit; Sec 6(c) it
  must carry an AFA-validated opt-out; Sec 7 post-debit notification also
  required; Sec 8(a) no OTP up to **Rs 15,000**; Sec 8(b) that ceiling rises to
  **Rs 1,00,000** for insurance, mutual funds and credit card bills
  (`requires_otp(amount, category=...)` — the generated population is entirely
  `"general"`, but the threshold is not hardcoded anywhere else).
- **NPCI UPI Autopay** — at most **4 attempts per cycle**; execution restricted
  to 00:00–07:00 IST (`in_execution_window`).
- **NACH** — return code **`04`** = Balance Insufficient; a failed NACH debit
  levies a **Rs 500** bounce charge on the customer. UPI Autopay failures do
  not. Never blend these into one per-attempt cost.
- **The six failure classes** are named here too, with `NEVER_RETRY_CLASSES`
  covering the two (expired card, opted out) where no date can ever succeed.
  Only `insufficient_funds` is generated today; the taxonomy exists so the
  router and class modules have one canonical spelling to import, and
  `FailureEvent` validates against it.

**Who sends the notification.** RBI places that duty on the **issuer** — the
customer's bank or UPI app — not on the merchant or its PSP. The sequencer
schedules the presentment; the issuer notifies and handles the opt-out. It
cannot suppress or reword the message, only decide whether to trigger one, which
is exactly why the attrition has to be priced rather than managed away.

### The opt-out model

`optout_probability` scales the customer's base sensitivity by three factors:
**annoyance** (×1.4 per extra notification in the same cycle), **tenure**
(long-tenure customers are stickier), and **recency**. Capped at 0.25.

**The recency direction is a switch, not a constant**, because it is the single
modelling choice most capable of inverting the project's conclusion:

- `"decay"` (default) — frustration is freshest right after the bounce and
  fades. Makes **waiting cheap, hammering expensive**.
- `"build"` — resentment accumulates while the debt drags on unresolved. Makes
  **waiting expensive, prompt resolution cheap**.

Both are plausible and neither is measured. They are exact reflections of each
other (their multipliers sum to a constant at every lag), so flipping
`direction=` is a clean ablation. **Every headline result must be reported under
both.** A conclusion that survives only one of them is a conclusion about
`RECENCY_WEIGHT`, not about retry policy.

---

## Assumptions worth knowing

These were choices the brief left open. Each is easy to change:

1. **`Customer.balance_seed`** is an extra hidden field beyond the specified
   model. It makes `daily_balance(customer, ...)` deterministic without
   threading a global RNG through the oracle. `agent_view` strips it.
2. **`cycle_start_date == original_debit_date`.** A monthly mandate's cycle runs
   debit-date to debit-date. The field is kept separate because retries are
   scoped to the cycle, not to the attempt.
3. **`success_probability_true` is unconditional** — it does not condition on
   the failures already observed on the customer's realised path. It answers
   "how often does a customer like this have the money on this date".
4. **Salary is forced to at least 8x the subscription amount**, so nobody puts a
   fifth of their income on one mandate. A side effect is that the Rs 5,000+
   tier goes to high earners and fails slightly *less* than the Rs 1,000–4,999
   tier.
5. **Historical opt-outs are not simulated.** Customers who would have cancelled
   during the observed 12 months are still present, so the population is not
   survivor-biased towards the insensitive.
6. **`monthly_ltv` is annual value** — it is `amount × 12`, named for the
   billing period rather than the horizon. `Customer.annual_ltv` is an alias
   that says so; prefer it in the objective function so the units cannot be
   misread.
7. **Income shocks are permanent within the window.** A real income interruption
   often ends after a few months. Modelling it as permanent is what guarantees
   CANCEL is unambiguously correct for that group; a recovering variant would
   make the stopping decision harder and is a reasonable next refinement.
8. **Only one failure class is generated.** `insufficient_funds` is the whole
   population. The other five are named in `rules.py` and validated against, but
   nothing emits them yet.

### Still open

- **Legacy recovery is 37.5 % mean but ranges 23.9–46.0 % across seeds.** The
  mean is on target; the per-seed spread is wide because it is a proportion over
  only ~70 failures. Larger `n` tightens it. Seed 42 specifically runs hot at
  47.2 %, so quote the multi-seed mean, not seed 42.
- **The `"build"` irritation direction is implemented but not yet evaluated.**
  The switch exists; the ablation has not been run.
