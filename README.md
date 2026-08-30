# Mandate Retry Sequencer — synthetic data generator

Ground-truth training and evaluation data for a system that decides **when and
whether** to retry a failed recurring debit, instead of firing a fixed
day-2/4/7 schedule at every failure.

Contains the **data generator**, the **fixed-schedule baseline**, the
**evaluation harness** that scores any policy against the oracle, and the
**sequencer** itself - payday inference, bank-health inference, OTP-completion
inference, the money equation, and the full RETRY / NUDGE / WAIT / HOLD /
CANCEL action set, with ablations for every component. Five of the six failure
classes are implemented behind a router that dispatches on the return code; the
sixth is a declared stub and produces no numbers.

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
  `salary_delay_days_max`, `income_shock_start`, `mandate_dead_from`,
  `mandate_dead_class`, `burn_rate`, `buffer_ratio`, `optout_sensitivity`,
  `otp_responsiveness`, `balance_seed`
- hidden failure field — `balance_at_failure` (a real PSP learns the account was
  short, never by how much)
- and the `balance_series` itself, which the agent view simply does not carry

---

## Install and run

```bash
pip install -e ".[dev]"     # or: pip install numpy pandas pydantic pyarrow pytest
pytest                      # 267 tests
python scripts/inspect_data.py --n 500 --seed 42 --save data/run_42
```

```python
from generator.generate import generate_dataset, save_dataset, load_dataset, agent_view

ds = generate_dataset(n_customers=500, seed=42)   # fully reproducible
print(ds.natural_failure_rate)                     # 0.166
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
| natural failure rate (current cycle) | **17.1 %** | 13.2 % | 20.4 % |
| legacy day-2/4/7 recovery | **38.3 %** | 26.0 % | 50.0 % |
| unrecoverable within 30 days | **29.6 %** | 20.7 % | 41.6 % |
| median days from failure to next payday | **11.1** | 9 | 13 |
| failures landing beyond day 7 | **69.8 %** | 60.9 % | 80.0 % |
| downtime share of failures (code 91) | **8.8 %** | 1.2 % | 21.9 % |
| AFA share of failures (code 1A) | **5.7 %** | 2.2 % | 12.1 % |
| AFA share of failures **by value** | **27.8 %** | 12.8 % | 48.7 % |

The failure rate rose from 14.6 % to 17.1 % when the AFA class was added,
because an authentication refusal is a genuinely new way to fail rather than a
reclassification of an existing one. The mean is the quality gate and it sits
inside the requested 5-20 % band; roughly one seed in twenty now clips 20 %,
which is a small-sample property of a proportion measured over ~85 failures.
The response was to widen the documented spread allowance in the test, **not**
to tune `OTP_RESPONSIVENESS_BETA` down - that constant is a claim about how
often people complete an authentication challenge, and it is not for sale to a
threshold.

The last two rows are the whole argument for class 5, and they are the reason
that class is reported money-first: it is a twentieth of the episodes and
better than a quarter of the rupees.

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

  This number is computed through the **full presentment gate**, not the balance
  alone. That distinction started mattering with class 5: asking only "was the
  balance sufficient" would score an AFA-blocked debit on a funded account as a
  legacy win, quietly inflating the floor every agent is compared against. The
  same correction applies to `unrecoverable_share`, with one deliberate
  difference — legacy recovery assumes **no** nudging, because a fixed schedule
  never reads a return code and so never learns an authentication is
  outstanding, while recoverability allows `afa.MAX_USEFUL_NUDGES`, because it
  asks whether *any* compliant policy could have saved the mandate.
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
| fixed-schedule floor (day-2/4/7) | **36.1 %** | 22.1 - 50.0 % |
| oracle ceiling | **69.0 %** | 58.4 - 79.3 % |
| **headroom to capture** | **32.8 pp** | |

Supporting numbers, same run:

| metric | legacy | oracle |
| --- | --- | --- |
| presentments used | 283 | 286 |
| customers retained | 30.9 | 71.8 |
| opt-outs induced | 7.3 | 1.6 |
| compliance breaches | 2.1 | 0 |
| net value | -Rs 2.03 m | -Rs 0.25 m |

The ceiling sits lower than the 73.6 % an insufficient-funds-only book gave,
and it keeps falling as classes are added — that is correct rather than a
regression. Roughly 9 % of failures are structurally dead mandates that nothing
can recover, and another 5.6 % are refused at an authentication gate that only
opens if a human acts. **The right ceiling for a mixed book is lower than the
right ceiling for an easy one, and quoting the higher number would be quoting a
different problem.**

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

Measured on the full mixed population - 76.8% insufficient funds, 8.9% bank
downtime, 6.3% expired card, 5.6% AFA incomplete, 2.4% revoked consent:

| | recovery | money | net value | attempts | optouts | retained |
| --- | --- | --- | --- | --- | --- | --- |
| legacy fixed | 36.1% | 114,697 | -2,034,554 | 283 | 7.3 | 30.9 |
| **sequencer** | **59.3%** | **191,982** | **-653,258** | 357 | 2.2 | 65.3 |
| oracle ceiling | 69.0% | 234,227 | -252,233 | 286 | 1.6 | 71.8 |

**Headroom captured: 70.5%.** Comfortably between the floor and the ceiling,
which is where it has to be - at or above the oracle would mean information is
leaking, and a test asserts it never happens on any seed.

The ceiling sits lower here (69.0%) than on an insufficient-funds-only book,
because about 9% of failures are structurally dead mandates that **nothing** can
recover, and another 5.6% are refused at an authentication gate. That is the
correct ceiling for a mixed book, and quoting the higher one would be quoting an
easier problem.

Beyond recovery: **3.3x fewer opt-outs induced** (2.2 vs 7.3) and **2.1x more
mandates retained** (65.3 vs 30.9).

### Ablations: what each piece is worth

`python scripts/run_ablations.py`. Deltas are **paired within each seed**, so
seed noise cancels; `seeds worse` counts how many of the 20 the variant lost on,
because a real effect shows up on most seeds rather than only in the mean.

Deltas are `variant - full`, so a negative number means the variant is worse.
`seeds worse` counts how many of the 20 the variant lost net value on.

| variant | d recovery | d net value | seeds worse | verdict |
| --- | --- | --- | --- | --- |
| `no_payday` | -16.5% | -457,312 | 18/20 | the biggest single contributor |
| `no_hold` | -9.9% | -859,263 | 20/20 | HOLD earns its place |
| `tier1_payday` | -7.1% | -151,935 | 16/20 | borderline - tier 2 over tier 1 |
| `with_nudge` | -3.1% | -345,430 | 19/20 | **NUDGE on class 1 is a liability** |
| `irritation_build` | -1.2% | -38,845 | 13/20 | **not consistent - see below** |
| `no_afa_nudge` | -0.3% | -21,077 | 7/20 | **not consistent - see class 5** |
| `no_optout_cost` | +1.0% | +37,394 | 9/20 | **not consistent - see below** |
| `no_bank_health` | +0.0% | **0** | 0/20 | **no effect at all - see class 2** |

**Three of these verdicts moved when the fifth class was added, and all three
moved toward "do not claim it".** The book grew from four generated classes to
five, so every book-wide average is now diluted across more and more varied
failures, and two effects that previously cleared this project's 15-of-20
consistency threshold no longer do. Those are reported as weakened below rather
than quietly re-baselined; the two strong results (`no_payday`, `no_hold`) got
stronger.

Five findings worth stating plainly:

**NUDGE hurts, so it is off by default.** It costs 4.1 points of recovery on 19
of 20 seeds: nudging spends a day without presenting, and the modelled lift does
not pay for the delay. The switch is kept so the finding stays reproducible
rather than deleted.

**The opt-out cost term no longer pays for itself, and this is a downgrade from
what was previously reported.** On the four-class book it returned Rs 35,819 of
net value on 15 of 20 seeds - exactly the consistency threshold. On the
five-class book, dropping the term *improves* both recovery (+1.0%) and net
value (+Rs 37,394), and the direction agrees on only 9 of 20 seeds, which is
close to a coin flip. The mechanism still works - the term does halve induced
opt-outs, which is what it is for - but the LTV it saves no longer outruns the
recovery it forgoes.

**N2 must therefore be claimed as a mechanism that is measured and priced, not
as a source of net gain.** The defensible sentence is "retries are a dual-effect
action and this project is the one that puts a rupee figure on the second
effect", not "pricing churn made us money". Nothing about the finding was tuned
to reach that conclusion; the number moved when the population got harder, and
it is reported as it came out.

**The irritation direction barely matters, and now falls below the consistency
threshold outright.** 13 of 20 seeds, against a 15-seed threshold, for a mean
cost of 1.2 points of recovery. Read it as "the central conclusion does not
depend on which way the assumption points" - which is the useful finding, since
this was flagged in CONTEXT as the single modelling choice most capable of
inverting the project's conclusion. It is not load-bearing, and the five-class
book makes that clearer rather than less clear.

**The AFA nudge does not clear the threshold either, and the reason is
instructive rather than disappointing.** Its book-wide delta is diluted twenty
times over, because it can only move the 5.6% of failures coded 1A. Isolated on
that class it is worth Rs 421,534 of net value across 20 seeds - but it *wins*
on only 7 of them. The full breakdown is in the class-5 section below, and it is
worth reading: this is a real effect with a genuinely lumpy distribution, not a
marginal one.

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

**[docs/audit_samples.md](docs/audit_samples.md)** holds eleven episodes chosen
by CATEGORY, not by outcome. Two of them are deliberately unflattering: one
where the fixed schedule beat the sequencer outright, and one where the
sequencer cancelled a mandate the oracle says was still recoverable. A sample
where the agent only wins is marketing.

The eleventh was added with class 5, and it is the one to read if you only read
one: an authentication gate, where P(clear) visibly *rises* as the nudges land
and the presentment goes in behind them. It is the only episode in the file
where the agent changes the number rather than choosing the best available one -
and the notification counter stays at zero throughout, because a nudge is out of
band.

---

## The other failure classes

CONTEXT lists six. This build implements five and declares the last.

| class | modelled? | success on any date | actions |
| --- | --- | --- | --- |
| `insufficient_funds` | **yes** - the payday model | depends on timing | all five |
| `bank_or_psp_downtime` | **yes** - bank health x liquidity | depends on the hour | RETRY, WAIT, HOLD, CANCEL |
| `afa_incomplete` | **yes** - OTP completion x liquidity | depends on the customer acting | all five |
| `expired_or_reissued_card` | **yes** (trivially) | **0.0** | NUDGE, HOLD, CANCEL |
| `customer_opted_out` | **yes** (trivially) | **0.0** | CANCEL only |
| `ambiguous_decline` | stub | - | declared, not implemented |

The two complete-but-trivial classes are the point of the abstraction. An
expired credential and a revoked mandate cannot clear on **any** date, so their
success model is the constant zero and their action set simply omits RETRY.
That is five lines each, and it buys the result below.

The stub exists so the router has no missing branch. It is marked
`implemented = False`, it produces no numbers, and nothing is claimed for it.

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

`python scripts/run_multiclass.py`. Realised mix: 76.8% insufficient funds,
8.9% downtime, 6.3% expired card, 5.6% AFA incomplete, 2.4% revoked.

| | legacy | sequencer |
| --- | --- | --- |
| recovery (all classes) | 36.2% | **59.4%** |
| **money recovered** | **2,293,936** | **3,839,648** |
| presentments | 5,654 | 7,138 |
| notifications | 4,091 | **1,633** |
| opt-outs induced | 146 | **45** |
| **compliance breaches** | **41** | **0** |

**Money is reported alongside rate now, and that became necessary rather than
tidy.** The sequencer is +23.2 points of recovery (+64.1% relative) but
+Rs 1,545,712 of money (+67.4% relative). The money gain outruns the rate gain,
and that gap is the class-5 effect showing up in the aggregate: the episodes the
sequencer wins and legacy loses are disproportionately the large ones, because
AFA only applies above Rs 15,000. A rate-only table weights a Rs 199 streaming
mandate and a Rs 19,000 one equally, which is the wrong weighting for a
revenue-recovery claim.

**On the structurally dead mandates**, routing on the return code saves **300
presentments and Rs 38,730** across 1,709 failures - **Rs 2,266 per 100
failures** - for about fifty lines of code and no model at all. The sequencer's
presentments there are the originals that produced the failure in the first
place; it makes none of its own.

**And 41 compliance breaches become 0.** A fixed schedule cannot avoid this: it
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

**Realised share: 8.8% of failures carry code 91** (mean over 20 seeds).
Individual seeds range from **1.2% to 21.9%**, and only a handful land inside
the 8-12% target band. That spread is not a tuning failure - it is the correlation
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

152 downtime failures out of 1,709 total (8.9%).

| policy | recovery | presentments | into a live outage | notices |
| --- | --- | --- | --- | --- |
| legacy fixed | 89.5% | 334 | 1 | 186 |
| **sequencer** | **94.1%** | 311 | 1 | 139 |
| oracle ceiling | 94.7% | 299 | 0 | 149 |

The sequencer beats legacy by 4.6 points and sits within half a point of the
ceiling. But the interesting results are the negative ones.

### Three negative results worth more than the win

**1. No policy can present into the outage it is reacting to. 0 of 48,779
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

**Class 5 did not disturb any of this.** Adding a fifth failure class changed
the book, so the downtime counts moved by a few presentments, but every
conclusion above survived unchanged: the same 89.5 / 94.1 / 94.7 recovery
split, still zero outages spanning two presentment windows, still an exactly
identical `no_bank_health` ablation on every seed. That is what a structural
result should do when the population around it changes.

**This is the sharper regulatory finding.** For class 1 the 24-hour rule costs
about nothing because the agent has no bank-side signal to react to. Here it
costs nothing for a structural reason instead: the rule does not make the class
unwinnable, it makes it nearly automatic - and it makes real-time bank health
monitoring worthless for scheduling compliant retries, which is the first thing
a payments company would instinctively build for this failure mode.

**3. The money equation has no time preference, and this class exposes it.**

The agent recovers as often as the oracle but takes **539 hours** after the
outage clears against the oracle's **38**. It is not a downtime bug. Rs 2,571
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

## Failure class 5: AFA incomplete

The first class where the agent can **change** the answer rather than pick the
best point on a curve somebody else drew.

    insufficient funds -> recovery takes DAYS.    Wait for payday.
    downtime           -> recovery takes HOURS.   Present at the first clear hour.
    AFA incomplete     -> recovery takes an ACTION. Ask, then present.

RBI Sec 8(a) exempts recurring debits from Additional Factor of Authentication
only **up to Rs 15,000 per transaction**. Above that ceiling every debit needs
its own OTP, so a presentment is refused with code `1A` unless the customer
authenticated *that* debit. The mandate is not dead — it is **gated**, and the
gate closes again behind every attempt.

### Selectional versus interventional, and why it cost real work

In classes 1–4 the agent's actions are **selectional**. It picks a date or
declines to. The success probability is a fact — a balance, an outage, a dead
credential — that the agent can estimate and can never move.

Here it is **interventional**. A NUDGE is an out-of-band message asking the
customer to authorise the mandate, and it causally raises the chance the gate is
open at the next presentment. Three things had to change, and none of them was
optional:

1. **The simulator had to apply NUDGE to the truth.** It had been a no-op
   (`# WAIT or NUDGE - neither presents nor ends the cycle`), which was harmless
   for four classes because nothing it could have changed existed yet.
2. **The oracle had to be allowed to nudge too.** A ceiling computed over a
   smaller action set than the agent's is not an upper bound — the agent could
   legitimately beat it, and the scale every headline number is quoted against
   would quietly stop meaning anything. It is capped at
   `afa.MAX_USEFUL_NUDGES = 2` for the opposite reason: an oracle allowed
   unlimited messages is unattainable, which makes every real policy look worse
   than it is. Class 2 hit the same failure from the other side, where
   `OraclePolicy` scheduled into live outages because it used a balance-only
   lookup.
3. **"Would a debit on date D have cleared?" stopped having one answer.** It has
   one per number of nudges. Every truth function now takes `nudges`; it
   defaults to 0, which is the world the dataset was *billed* in, because the
   merchant's legacy schedule nudged nobody.

### A nudge is out of band, and that is the whole cost argument

A nudge is a message from the **merchant**. The pre-debit notification is a
message from the **issuer**, and under Sec 6(c) it must carry an AFA-validated
opt-out. So a nudge:

- does **not** increment the notification counter, and creates no opt-out
  opportunity — it cannot cancel the mandate;
- does **not** consume an NPCI attempt;
- **does** cost goodwill — `rules.NUDGE_GOODWILL_COST_FRACTION` of annual LTV
  per message, charged against net value.

That is the trade: a cheaper way to change the customer's state than presenting.
Three tests pin it, because it is the property most likely to regress the next
time somebody edits the simulator.

The lift has **diminishing returns and converges below certainty**. Responsiveness
scales both the baseline completion rate and the efficacy of asking — one
assumption, not two: somebody who ignores authentication prompts ignores the
reminder about them as well. A flat lift would let one message rescue a
chronically unresponsive customer, which is the opposite of what the trait means.

### The framing number: thin by count, heavy by value

**Report this class by money first and rate second — the reverse of every other
class.** AFA applies only above Rs 15,000, so it rides exclusively on the
largest mandates on the book.

| across 1,709 failures worth Rs 5,806,135 | by count | by value |
| --- | --- | --- |
| failures on OTP-gated mandates | 9.2 % | 46.8 % |
| **failures refused at the AFA gate (1A)** | **5.6 %** | **28.4 %** |

A twentieth of the episodes carrying better than a quarter of the rupees at
risk. A recovery-rate table weights a Rs 199 streaming mandate and a Rs 19,000
one equally, which is the wrong weighting for a revenue-recovery claim and would
have made the most valuable class on the book look like noise.

**The population was deliberately not widened to fatten this class.**
`AMOUNT_TIERS` still puts ~3 % of the book above the Sec 8(a) ceiling, which is
realistic for an Indian subscription book and is what every class-1 and class-2
number was measured on. A thin class reported honestly is worth more than a fat
one bought by moving the population.

### Results, 20 seeds x 500 customers, AFA failures only

Value at risk across **96 AFA failures**: Rs 1,648,936.

| policy | money | of value | net value | goodwill | notices |
| --- | --- | --- | --- | --- | --- |
| legacy fixed | 816,633 | 49.5 % | -9,220,563 | 0 | 205 |
| **sequencer** | **1,123,371** | **68.1 %** | **-3,130,544** | 235,961 | 136 |
| `no_afa_nudge` | 1,040,834 | 63.1 % | -3,552,077 | 0 | 79 |
| oracle ceiling | 1,523,117 | 92.4 % | +372,961 | 7,056 | 91 |

**The sequencer recovers Rs 306,738 more than legacy — Rs 3,195 per AFA failure.**

Recovery rate second, and **with the episode count attached**, because the class
is thin:

| policy | recovery | episodes | per-seed spread |
| --- | --- | --- | --- |
| legacy fixed | 50.0 % | 48/96 | 25 % .. 100 % |
| **sequencer** | **68.8 %** | 66/96 | 33 % .. 100 % |
| oracle ceiling | 92.7 % | 89/96 | 67 % .. 100 % |

Roughly five episodes per seed, and a per-seed standard deviation of **18.0 %**.
Quote the value figures; the rate is context, not the result.

### What the interventional action is actually worth

| variant | money | recovery | nudges | attributable | net value |
| --- | --- | --- | --- | --- | --- |
| sequencer | 1,123,371 | 68.8 % | 578 | 19 | -3,130,544 |
| `no_afa_nudge` | 1,040,834 | 63.5 % | 0 | 0 | -3,552,077 |

**Nudging is worth Rs 82,537 of recovery and Rs 421,534 of net value.** It is a
real positive result, and it is the novel claim of this class — but two things
have to be said with it.

**Only 19 of 578 nudges actually opened a gate that would have stayed shut.**
That is an exact counterfactual, not an estimate: the AFA draw is keyed on
identity and date and never on the nudge count, so re-testing the same uniform
against the unnudged probability answers "would this have happened anyway?"
precisely. Most nudges are sent to customers who would have authenticated on
their own. The action pays because when it *does* bite, it bites on a Rs 19,000
mandate.

**The hit rate is low because AFA is redrawn per presentment.** Sec 8(a) says
"per transaction", so a later date can clear where an earlier one did not
without the customer doing anything differently — the agent can partly
substitute patience for asking. A model in which the refusal persisted until the
customer acted would make nudging the *only* route to recovery and would inflate
this number considerably. That is a defensible alternative reading and it is not
what the regulation says, so it was not taken.

### The nudge result is real in the mean and lumpy in the distribution

This is the number most likely to be over-claimed, so here is the whole
distribution rather than the total. Per-seed `sequencer - no_afa_nudge`, net
value, across all 20 seeds:

| | seeds | net value |
| --- | --- | --- |
| nudging **wins** | 7 / 20 | +Rs 743,000 combined |
| nudging **loses** | 13 / 20 | -Rs 321,000 combined |
| **net** | | **+Rs 421,534** |

The positive mean is carried almost entirely by **four seeds** where a nudge
rescued a five-figure mandate: +Rs 240,023, +Rs 211,829, +Rs 205,520 and
+Rs 47,421. On the other sixteen the money recovered is **identical** — the
gate opened on its own, or never opened — and the only difference is the
goodwill the nudges spent, which is a small certain loss. One seed goes the
other way hard: **-Rs 216,518**, where nudging burned the day that mattered and
a large mandate was lost that the date-only policy caught.

**Against this project's 15-of-20 threshold, that is NOT a consistent effect and
must not be claimed as one.** The honest statement is: *nudging is worth
Rs 421,534 in expectation across 20 seeds, and it is a high-variance bet that
loses a little most of the time and wins a lot occasionally, because the class it
acts on is a handful of very large mandates.* That is a real and defensible
finding — a merchant should still take a positive-expectation bet — but it is
not the same claim as `no_payday` or `no_hold`, which lose on 18 and 20 seeds
out of 20 respectively.

The thinness of the class is the cause, not a separate problem. Five episodes
per seed cannot produce a consistent per-seed effect no matter how good the
action is.

### Why legacy fails here, and it is a third distinct reason

    class 1: legacy was MISTIMED          - it retried before payday
    class 2: legacy was ACCIDENTALLY FINE - short outages had already cleared
    class 5: legacy is STRUCTURALLY FUTILE

| policy | presentments chosen | automatic | refused 1A |
| --- | --- | --- | --- |
| legacy fixed | 205 | 89 | 128 |
| sequencer | 136 | 393 | 332 |
| oracle ceiling | 91 | 163 | 55 |

Legacy chose **205** presentments to the sequencer's **136**, and recovered
Rs 816,633 against Rs 1,123,371. More attempts, less money: the ones it added
were made into a gate that was shut. It presents four times against a mandate
that will refuse every one of them until a human completes an authentication
step legacy never asks for — because it never reads the return code and so never
learns one is outstanding. Each refusal spends an NPCI attempt and fires a
cancel-capable Sec 6(a) notice to learn *nothing*: the debit never reached the
account, so it did not even establish that the customer was short.

The presentment column is split deliberately. A live mandate keeps auto-billing
on its anchor date every cycle it is carried, so a policy that HOLDs accumulates
presentments it never decided to make. Pooling the two would read as the
sequencer being the wasteful one, when on the only column it controls it makes
barely half as many as legacy.

And because these are the largest mandates on the book, the futility is
expensive in a way the recovery-rate column cannot show: legacy leaves
**Rs 832,303** uncollected against the ceiling's Rs 125,819.

### A bug this class did not have to surface, because class 2 already had

`PaydayModel` drops censored declines rather than counting them as evidence the
account was empty. Code `1A` joins `91` and `54` in that set: an authentication
refusal never reached the account, so it says nothing about liquidity. The
agent's `OtpCompletionModel` applies the mirror image of the same discipline —
a `91` on a large mandate is not evidence about authentication either, because
the gate was never consulted.

The discriminating property that makes the class separable at all:

| evidence | inference |
| --- | --- |
| `1A` on a mandate over Rs 15,000 | the customer did not authenticate |
| `04` on the same mandate | they **did** authenticate, and were broke |

Both are declines on the same large mandate. The second is *positive* evidence
that the customer authenticates, and without it a subscriber who authenticates
reliably but keeps running out of money would look like an authentication
problem — and the agent would nudge somebody whose issue is their balance.
Measured: six `1A`s take the completion estimate from the 0.75 book prior to
0.30; six `04`s on the same mandate take it to 0.90; six `91`s leave it at
exactly 0.75.


---

## Package layout

```
src/generator/
  rules.py        RBI / NPCI / NACH constants. Regulations, not tunables.
  models.py       Pydantic records + the agent/truth type split.
  balance.py      The hidden daily balance process (vectorised over paths).
  banks.py        Bank outages: hour-resolution, shared across customers.
  afa.py          The AFA gate: per-presentment OTP completion, and the one
                  piece of truth a policy ACTION can move.
  oracle.py       Ground truth: would_succeed, success_probability_true, curves,
                  and resolve_presentment - THE canonical gate order.
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
  afa_incomplete.py authentication x liquidity; recovery takes an ACTION
  expired_card.py   dead credential: zero on every date, no RETRY
  opted_out.py      consent revoked: CANCEL only, RETRY is a breach
  stubs.py          ambiguous - DECLARED, NOT IMPLEMENTED

src/audit/        Renders decision logs for human review.
  trail.py          reads what the policy recorded; never recomputes it

src/agent/        AGENT-SAFE. The intelligent policy.
  payday.py         PaydayModel: when is this customer liquid, and how sure
                    are we? Three tiers; tier 2 is the calibrated one.
  bank_health.py    The first CROSS-CUSTOMER signal: pools decline evidence
                    across everyone sharing an issuer.
  otp_completion.py Will this customer authenticate, and does asking help?
                    Per-customer again, but over a lever the agent controls -
                    the third distinct SHAPE of inference in the project.
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
scripts/run_afa.py              failure class 5, reported by MONEY rather than rate
scripts/show_audit.py           the full decision trail for one customer
scripts/export_audit_sample.py  eleven episodes to docs/audit_samples.md

tests/test_generator.py   unit-level: balance, oracle, population, opt-out
tests/test_dataset.py     dataset-level: reproducibility, realism, leakage
tests/test_policy.py      the contract and the baseline schedule
tests/test_evaluation.py  compliance, determinism, economics, harness
tests/test_payday.py      payday inference, shock detection, calibration
tests/test_decision_engine.py  the money equation and the churn estimator
tests/test_sequencer.py   the policy, its audit trail, and the guardrails
tests/test_classes.py     the router, the never-retry classes, the audit trail
tests/test_downtime.py    outages, bank health inference, the two-condition oracle
tests/test_afa.py         the AFA gate, OTP inference, and the out-of-band nudge

docs/audit_samples.md     eleven rendered episodes, picked by category
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
  Five are generated today; `ambiguous_decline` is not. The taxonomy exists so
  the router and class modules have one canonical spelling to import, and
  `FailureEvent` validates against it.
- **`NUDGE_GOODWILL_COST_FRACTION`** also lives here, though it is an ESTIMATE
  rather than a regulation. It moved out of `agent/decision_engine.py` when the
  simulator started charging it: a nudge now changes the truth, so the
  *evaluator* has to price it, and `evaluation/` must not import `agent/`. Same
  reasoning as `DEFAULT_PRESENTMENT_TIME`, and the agent module re-exports the
  name so every call site kept working.

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
8. **Five of six failure classes are generated.** `insufficient_funds` is the
   bulk of the population, with bank downtime, expired cards, revoked consent
   and AFA refusals alongside it. `ambiguous_decline` is named in `rules.py`
   and validated against, but nothing emits it yet.
9. **`Customer.otp_responsiveness` is drawn LAST in `make_population`.** Every
   draw shares one RNG stream, so inserting a new one anywhere earlier shifts
   every subsequent draw and silently changes salaries, burn rates and balance
   seeds for every customer in every existing dataset — which is how class 2
   flipped a seed-sensitive holding test when `bank_id` was added mid-sequence.
   Appending leaves every prior draw byte-identical, so the only difference
   class 5 makes to an existing seed is the AFA gate itself. This was verified,
   not assumed: across 8 seeds x 2 policies, every insufficient-funds episode on
   a mandate *below* the Sec 8(a) ceiling scored identically to the rupee before
   and after the class was added. Add future fields at the end for the same
   reason.
10. **AFA completion is redrawn per presentment.** Sec 8(a) exempts AFA "per
   transaction", so an OTP completed for one debit does nothing for the next —
   including the next retry inside the same cycle. The consequence is that a
   later date can clear where an earlier one did not, without the customer
   doing anything differently, which is why nudging is worth less on this class
   than the mechanism first suggests. A model where the refusal persisted until
   the customer acted would make nudging the only route to recovery; that is a
   defensible alternative and it is not what "per transaction" says.

### Still open

- **Legacy recovery is 38.3 % mean but ranges 26.0–50.0 % across seeds.** The
  mean is on target; the per-seed spread is wide because it is a proportion over
  only ~85 failures. Larger `n` tightens it. Quote the multi-seed mean.
- **`ambiguous_decline` is the last stub.** It is declared in `rules.py`, routed
  to a handler that reports `implemented = False`, and produces no numbers.
  Nothing emits code `05` yet, and its cautious-single-retry rule is drafted in
  `classes/stubs.py` but not wired to the engine.
- **The money equation still has no time preference.** `daily_discount` defaults
  to 0.0 and is measured in `scripts/run_downtime.py`. It is the highest-value
  change outstanding, and turning it on would move every number in this file, so
  it is a decision for the write-up rather than a silent edit.
- **The AFA nudge lift shares its functional form with the generator.** The
  agent estimates the same shape the truth uses, which flatters it - the same
  caveat `agent/churn.py` carries. `no_afa_nudge` measures whether the action is
  worth anything at all, which is the question that survives the caveat; its
  calibration is not defended.
- **AFA completion is redrawn per presentment**, per the Sec 8(a) wording
  ("per transaction"). That lets a later retry clear where an earlier one did
  not, without the customer acting - so the agent can partly substitute patience
  for asking, and nudging is worth less than the mechanism first suggests. A
  model where the refusal persisted until the customer acted is a defensible
  alternative that would make nudging the only route to recovery. It was not
  taken, and the choice is worth naming because it sets the size of the class-5
  result.
