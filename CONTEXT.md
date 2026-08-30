# CONTEXT — Mandate Retry Sequencer

Read this before touching code. It explains why the project is shaped
the way it is. The code answers "what"; this file answers "why".

## Submission target

Razorpay AI Buildathon, Track 3 (AI Revenue Recovery).
Direction: mandate retry sequencer. Deadline 3 September 2026.

Razorpay's stated bar for this track:
> "Show measured money recovered across a batch, with compliant
> escalation, stopping rules, and an audit trail."

Four nouns. Every design decision serves one of them.

## The problem in one paragraph

An Indian subscription debit fails. Almost every merchant responds with
a fixed schedule: retry day 2, day 4, day 7, then cancel. Same response
regardless of why it failed. This is wrong for all six failure classes,
but it is worst for insufficient funds, which is 60-70% of NACH returns
and 56% of merchant-initiated transaction failures.

## Why fixed retry fails specifically

Indian salaries cluster at month-end. Merchants set the debit day from
the signup date, which is uniform across days 1-28. So the debit day
carries no information about payday. A customer paid on the 7th and
debited on the 2nd will fail, and day-2/4/7 retries land on the 4th,
6th and 9th. Only the last one has a chance, and only by accident.

## The six failure classes

This project handles all six. They share one decision engine and differ
only in their success model and permitted actions.

| Class | Predicts success | Best action | Retry ever right? |
|---|---|---|---|
| Insufficient funds | Payday timing | RETRY after salary, or HOLD | Yes, if timed |
| Bank/PSP downtime | Bank health signal | RETRY within hours | Yes, quickly |
| Expired/reissued card | Nothing, mandate is dead | NUDGE to update card | Never |
| Customer opted out | Nothing, consent withdrawn | STOP | Never |
| AFA incomplete (>Rs 15,000) | Will they complete OTP | NUDGE to complete | Only after OTP |
| Ambiguous decline | Unknown | One cautious retry | Once, then stop |

Two of six should never be retried at all. Fixed schedules retry them
anyway. That is free recovery for a class-aware system.

## Selectional versus interventional classes

Worth naming, because it is the sharpest architectural distinction in the
project and it only became visible when class 5 was built.

In insufficient funds, downtime, expired card and revoked consent the agent's
actions are SELECTIONAL. It picks a date, or declines to pick one. The success
probability is a fact about the world - a balance, an outage, a dead
credential - that the agent can estimate but can never move. Its entire skill
is choosing well on a curve somebody else drew.

AFA incomplete is INTERVENTIONAL. Above the Sec 8(a) ceiling a debit clears
only if the customer authenticated it, and a NUDGE causally raises the chance
they did. The agent can move the curve, not merely pick a point on it.

Three consequences, all of which cost real work:

1. The simulator had to apply NUDGE to the truth rather than log it and move
   on. Through class 4 it was a no-op, which was harmless because nothing it
   could have changed existed yet.
2. The ORACLE had to be allowed to nudge too. A ceiling computed over a smaller
   action set than the agent's is not an upper bound, and the agent could
   legitimately beat it - which would silently destroy the scale every headline
   number is quoted against.
3. "Would a debit on date D have cleared?" stopped having one answer. It has
   one answer per number of nudges, so every truth function is parameterised by
   that count. Zero is the world the dataset was billed in; the ceiling is
   allowed two.

The three implemented models are also three different SHAPES of inference,
which is worth saying out loud because it looks like repetition and is not:

    payday.py         one customer in isolation - pooling would be meaningless
    bank_health.py    pooled across customers on one issuer
    otp_completion.py one customer again, but over an action we control

## Regulatory constraints (hard, not tunable)

RBI Digital Payments E-mandate Framework, 2026 (RBI/DPSS/2026-27/396,
21 April 2026):
- Sec 6(a): pre-debit notification at least 24 hours before EVERY debit
- Sec 6(c): that notification must offer an AFA-validated opt-out
- Sec 7: post-debit notification required
- Sec 8(a): no AFA required up to Rs 15,000 per transaction
- Sec 8(b): Rs 1,00,000 for insurance, mutual funds, credit card bills

NPCI UPI Autopay:
- Maximum 4 attempts per cycle (1 original + 3 retries)
- Execution restricted to the 00:00-07:00 non-peak window

NACH:
- Return code 04 = Balance Insufficient
- Failed NACH debit triggers a customer-side bounce charge (~Rs 500)
- Failed UPI Autopay debit carries no equivalent charge

All of these live in src/generator/rules.py. Read from there, never
hardcode.

## The decision object

State: attempts used, amount, days since failure, inferred payday
posterior, payment history, tenure, LTV, notifications sent this cycle,
rail.

Actions: RETRY(date), NUDGE, WAIT, HOLD, CANCEL

HOLD is the action fixed schedules lack. It means: skip this cycle,
keep the mandate alive, re-present next cycle at a payday-aligned date.
For insufficient funds this is frequently the correct terminal action,
because the mandate is structurally alive and the customer never chose
to leave.

Objective, per mandate:

    value = (P(success) * amount)
          - cost_of_attempt
          - (P(optout) * customer_LTV)

Stopping rule: continue only while the best schedulable date produces a
positive value.

## Why the opt-out term exists

RBI Sec 6(a) and 6(c) together mean every scheduled retry obligates
delivery of a message to the customer saying "we will debit you
tomorrow, tap here to stop it." The customer is by construction short
of money and possibly irritated.

So a retry is a dual-effect action: it can recover the payment AND it
can destroy the mandate. Recovering Rs 199 while losing Rs 2,400 of
annual LTV is a net loss.

No US or EU retry engine models this because no US or EU regime
mandates the notification. This is the most India-specific and most
defensible part of the project.

## The 24-hour commitment lag

Because notification must precede debit by 24 hours, the agent cannot
observe a salary credit and react. It must commit to a retry date at
least a day ahead using only information available at commit time.

This makes the policy a scheduler under a lead-time constraint, not a
reactive controller. It is also measurable: see Novelty 3.

## Four novelty claims

N1. Time-to-liquidity as the primitive.
Model P(success | date) as a hazard over an inferred salary-credit
posterior, not as a function of retry index. Published industry work
ranks time slots: Dropbox divides an 8-day window into 192 one-hour
chunks and picks the top-ranked one. Stripe and Recurly describe
similar approaches. None publish a cost model or stopping rule.
Grounding: arXiv 1806.05362 (predicting balances from recurring salary
transactions); PyDTS, arXiv 2204.05731 (discrete-time competing risks).

N2. Retry priced as a dual-effect action.
The opt-out hazard above, quantified in rupees and subtracted before
deciding. India-specific.

N3. Measuring the cost of the 24-hour rule.
Run the agent twice: once with the real lead time, once with a
zero-lag oracle. The gap is the rupee cost of compliance. Unpublished
number, cheap experiment, directly relevant to Razorpay.

N4. A stopping rule with a finite-sample guarantee.
Calibrate the stop threshold via conformal risk control so a
business-defined risk is provably controlled, rather than tuned.
Grounding: Angelopoulos et al., Conformal Risk Control (arXiv
2208.02814); Candes, Lei & Ren, Conformalized Survival Analysis
(JRSS-B 2023) for calibrated lower bounds under censoring.

Closest academic precedent for the overall shape: Abe et al.,
"Optimizing Debt Collections Using Constrained Reinforcement Learning"
(KDD 2010), a constrained MDP deployed at the New York State
Department of Taxation and Finance. It supplies the batch-budget
machinery but has neither a commitment lag nor an action-induced
attrition hazard.

## The ground-truth oracle

The generator invents each customer's salary date, burn rate, cushion
and shock process, so the daily balance is a complete series covering
every day, including days nobody debited on.

That means "would a debit of Rs X on date D have succeeded?" has an
exact answer for every D. Not a prediction. The answer.

A real PSP cannot do this. If they retried on day 2 and it failed, they
will never know whether day 5 would have worked. Comparing two policies
on real data requires running both live for months.

This makes exact counterfactual policy evaluation possible, and it is
the strongest thing to lead with when presenting.

## Honesty items

These go in the failure list. Judges are payments experts and will
find them anyway.

1. Per-attempt failure cost differs by rail. UPI Autopay failures cost
   the merchant almost nothing. NACH failures trigger a real bank
   bounce charge on the customer (~Rs 500, varies by bank). Never use a
   single blended number. Parameterise by rail and state the source.

2. The agent does not send the pre-debit notification. RBI places that
   duty on the issuer, meaning the customer's bank or UPI app. The
   agent schedules the presentment; the issuer sends the notification
   and handles the opt-out. Claiming otherwise reveals a
   misunderstanding of the payment flow.

3. The irritation-decay direction is an assumption and nothing in the
   data can settle it. The generator defaults to frustration decaying
   after the bounce, which makes waiting cheap and hammering expensive.
   Resentment building while the debt drags on unresolved is equally
   plausible, and nobody has measured which one is real. Because it is
   the single modelling choice most capable of inverting the project's
   conclusion, it is a switch rather than a constant
   (IrritationDirection in generator/optout.py, threaded through every
   function there) and every headline number is reported under both
   directions via the irritation_build ablation. The gap is small:
   -1.3% recovery on 16 of 20 seeds, against the 15-seed threshold this
   project uses to call an effect consistent. Read that as "the
   conclusion does not lean on the assumption", not as the assumption
   having been validated.

4. Adding the fifth class moved the book-wide failure rate from 14.6% to
   17.1% across 20 seeds, because AFA is a genuinely new way to fail rather
   than a reclassification of existing failures. Both numbers are inside the
   5-20% band the brief asks for, and the mean is the gate; the worst
   individual seed now clips 20% on roughly 1 seed in 20, which is a
   small-sample property of a proportion over ~70 failures. The response was
   to widen the documented spread allowance in tests/test_dataset.py, NOT to
   tune OTP_RESPONSIVENESS_BETA down. That constant is a claim about how often
   people complete an authentication challenge and is not for sale to a
   threshold.

   Two knock-on numbers moved with it and both are now measured through the
   full presentment gate rather than the balance alone: legacy recovery
   (37.5% -> 38.9%) and unrecoverable share. The old code asked "was the
   balance sufficient", which would have scored an AFA-blocked debit on a
   funded account as a legacy win and quietly inflated the floor every agent
   is compared against.

5. The bank concentration table is realistic in SHAPE only. Per-issuer
   UPI Autopay mandate share is not published by NPCI. What is public is
   the broad concentration: a handful of large issuers carry most
   mandates, with a long tail of smaller banks and small finance banks.
   The weights in src/generator/banks.py reproduce that shape and
   nothing more. Never quote them as facts about any named bank.

## Architecture for merging the six classes

    core/
      decision_engine.py    money equation, shared
      actions.py            the five moves, shared
      constraints.py        24h rule, attempt caps, shared

    classes/
      insufficient_funds.py payday model
      downtime.py           bank health model
      expired_card.py       returns "never retry"
      opted_out.py          returns "never retry"
      afa_incomplete.py     OTP completion model
      ambiguous.py          cautious single retry

    router.py               reads return code, dispatches to class
    evaluator.py            shared

Every class file exposes exactly two functions:

    def success_probability(customer, date) -> float
    def allowed_actions(state) -> list

The engine calls those two and does not care which class it is
handling.

Build order: insufficient funds fully, then expired card and opted out
(twenty minutes each, they just return "never retry"), then downtime,
AFA, ambiguous, then the router. After the first three you have half
the classes covered for barely more than the effort of one.

## Evaluation rules

- Every claim must have a number
- Every number must come from a reproducible run
- Fixed seeds everywhere
- Never report a single run; report mean and spread across 20 seeds
- The baseline obeys the same regulatory constraints as the agent.
  Crippling it invalidates the comparison.

Metrics to report: money recovered, recovery rate, attempts used, cost
per rupee recovered, wasted attempts (under 20% success chance),
wrongly cancelled customers, customers retained.

Required experiments: ablations (drop payday model, drop opt-out cost,
drop HOLD), the 24-hour lag ablation, and 20-seed robustness.

## Open issues

The three issues that stood here through class 1 - baseline recovering
63.8%, no unrecoverable customers, irritation decay hardcoded - are all
closed. Git history has what they said.

1. Downtime failure share is volatile by nature, and that is the class
   behaving correctly rather than a tuning failure. Mean over 20 seeds is
   9.1% of failures carrying code 91, but individual seeds run from 1.3%
   to 22.3% and only 3 of 20 land inside the 8-12% target band.
   Insufficient funds is an independent per-customer draw, so its share
   is stable; one outage fails every mandate on that issuer at once, so a
   large bank down on a busy debit day swings a whole seed. Pinned as a
   property in tests/test_dataset.py rather than tuned away, because
   narrowing the spread would mean removing the cross-customer
   correlation that makes the class worth modelling at all.

2. The money equation has no time preference. Nothing penalises delay, so
   Rs 2,571 collected in thirty days outscores Rs 2,551 collected
   tomorrow. On the downtime class the agent recovers as often as the
   oracle but resolves 538 hours after the outage clears, against the
   oracle's 34. DecisionEngine.daily_discount prices this and defaults to
   0.0. Measured on downtime, 1%/day roughly halves the mean post-clear
   delay (23.2 -> 10.7 days) at no cost in recovery (96.1% either way).
   It is left off so no class-1 number already reported changes silently;
   switching it on is a decision for the write-up, not a quiet edit. It
   is the highest-value change outstanding.

3. Class 6 (ambiguous) is still a stub in src/classes/stubs.py. It declares
   implemented = False, returns a success probability of 0.0, and is excluded
   from every reported number. It needs its cautious-single-retry rule wired
   through to the engine. Class 5 (afa_incomplete) is now built; the stub
   module holds only the one class.

4. The AFA class is THIN by episode count - 5.6% of failures across 20 seeds,
   roughly five episodes per seed - so its recovery RATE carries a wide
   interval and should never be quoted without the episode count beside it.
   Its VALUE figures are what the class is for: those same failures carry
   28.4% of the rupees at risk, because AFA applies only above Rs 15,000 and
   so rides on the largest mandates on the book. Reported money-first in
   scripts/run_afa.py for that reason, which is the reverse of every other
   class. Widening AMOUNT_TIERS would fatten the class and is deliberately NOT
   done: a thin class reported honestly is worth more than a fat one bought by
   moving the population every earlier number was measured on.

5. The AFA nudge lift has the same functional form in the generator and in the
   agent. That flatters the agent - on real data a merchant would fit the
   dunning response from its own history and get the shape somewhat wrong -
   and it is the same caveat agent/churn.py already carries. What survives it
   is the no_afa_nudge ablation, which measures whether the action is worth
   anything at all rather than whether the lift is calibrated.
