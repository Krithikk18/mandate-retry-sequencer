# Audit samples

Rendered by `scripts/export_audit_sample.py`. Every number below is
read from what the policy recorded at decision time - the renderer
does not recompute anything.

These are picked by CATEGORY, not by outcome. Two of them are cases
where the agent did worse than the fixed schedule or gave up on a
recoverable mandate; a sample where the agent only wins would not be
worth reading.

| # | case | outcome | agent net | legacy net |
| --- | --- | --- | --- | --- |
| 1 | A clean recovery | RECOVERED | Rs 570 | Rs 570 |
| 2 | Where the sequencer earns its keep | RECOVERED | Rs 298 | Rs -5,384 |
| 3 | Where the fixed schedule beat the sequencer | CHURNED | Rs -2,775 | Rs -2,756 |
| 4 | A HOLD: nothing this cycle is worth presenting on | HELD | Rs -7,420 | Rs -10,328 |
| 5 | A correct CANCEL on a customer whose income stopped | CANCELLED | Rs -3,599 | Rs -5,144 |
| 6 | A CANCEL that was wrong | CANCELLED | Rs -4,823 | Rs -5,338 |
| 7 | An opt-out on the pre-debit notification | OPTED_OUT | Rs -45,622 | Rs -45,617 |
| 8 | A held mandate that drifted away | CHURNED | Rs -1,477 | Rs -1,328 |
| 9 | A NACH mandate, where a failed retry costs the customer Rs 500 | RECOVERED | Rs 278 | Rs -752 |
| 10 | A long wait for a distant payday | RECOVERED | Rs -1,148 | Rs -11,048 |

## 1. A clean recovery

The ordinary case: infer the payday, price the dates, present once on the best one.

`--customer CUST_0009 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0009 · Rs 580 · upi_autopay · 3 months · LTV Rs 6,960
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

08 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

08 Jul   DECISION: WAIT  (unchanged for 6 days)
         Payday posterior: day 9 (conf 0.42)
         Shock probability: 0.05
         Best candidate 15 Jul scored +Rs 257
           P(clear) 0.80 x Rs 580            =    +Rs 465
           processing                        =      -Rs 5
           bounce risk 0.20 x Rs 0           =       Rs 0
           opt-out risk 0.029 x Rs 6,960     =    -Rs 202
         Runners-up: 14 Jul +Rs 257 · 16 Jul +Rs 256 · 13 Jul +Rs 255
         best date is not yet due for commitment

14 Jul   DECISION: RETRY
         Payday posterior: day 9 (conf 0.42)
         Shock probability: 0.05
         Best candidate 15 Jul scored +Rs 257
           P(clear) 0.80 x Rs 580            =    +Rs 465
           processing                        =      -Rs 5
           bounce risk 0.20 x Rs 0           =       Rs 0
           opt-out risk 0.029 x Rs 6,960     =    -Rs 202
         Runners-up: 16 Jul +Rs 256 · 17 Jul +Rs 253 · 18 Jul +Rs 249
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

15 Jul   PRESENTMENT - SUCCEEDED, Rs 580 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 570 · 7 days to resolution
Mandate retained · LTV preserved Rs 6,960
legacy_fixed on the same customer: presented 10 Jul; recovered, 1 notifications sent, net +Rs 570
══════════════════════════════════════════════════════════════════════
```

## 2. Where the sequencer earns its keep

The fixed schedule spent its whole budget and lost the mandate; the sequencer waited for payday.

`--customer CUST_0016 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0016 · Rs 447 · upi_autopay · 5 months · LTV Rs 5,364
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

21 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

21 Jul   DECISION: HOLD
         Payday posterior: day 1 (conf 0.40)
         Shock probability: 0.87
         Best candidate 07 Aug scored -Rs 72
           P(clear) 0.12 x Rs 447            =     +Rs 52
           processing                        =      -Rs 5
           bounce risk 0.88 x Rs 0           =       Rs 0
           opt-out risk 0.022 x Rs 5,364     =    -Rs 119
         Runners-up: 06 Aug -Rs 72 · 08 Aug -Rs 72 · 05 Aug -Rs 73
         hold value        -Rs 3,128
         cancel value      -Rs 5,364
         income shock inferred

--- cycle 2 ---

21 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Aug   DECISION: HOLD
         Payday posterior: day 1 (conf 0.38)
         Shock probability: 0.87
         Best candidate 07 Sep scored -Rs 72
           P(clear) 0.12 x Rs 447            =     +Rs 52
           processing                        =      -Rs 5
           bounce risk 0.88 x Rs 0           =       Rs 0
           opt-out risk 0.022 x Rs 5,364     =    -Rs 119
         Runners-up: 06 Sep -Rs 72 · 08 Sep -Rs 73 · 05 Sep -Rs 73
         hold value        -Rs 3,128
         cancel value      -Rs 5,364
         income shock inferred

--- cycle 3 ---

21 Sep   PRESENTMENT - SUCCEEDED, Rs 447 recovered
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 298 · 62 days to resolution
Carried 2 cycles at Rs 134 in carrying cost
Mandate retained · LTV preserved Rs 5,364
legacy_fixed on the same customer: presented 23 Jul, 25 Jul, 28 Jul; cancelled, 3 notifications sent, net -Rs 5,384
══════════════════════════════════════════════════════════════════════
```

## 3. Where the fixed schedule beat the sequencer

This is here on purpose. A sample where the agent only ever wins is curated, and a reviewer should assume the worst of it.

`--customer CUST_0003 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0003 · Rs 228 · upi_autopay · 9 months · LTV Rs 2,736
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

19 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

19 Jul   DECISION: HOLD
         Payday posterior: day 1 (conf 0.44)
         Shock probability: 0.87
         Best candidate 06 Aug scored -Rs 27
           P(clear) 0.12 x Rs 228            =     +Rs 27
           processing                        =      -Rs 5
           bounce risk 0.88 x Rs 0           =       Rs 0
           opt-out risk 0.018 x Rs 2,736     =     -Rs 49
         Runners-up: 05 Aug -Rs 27 · 07 Aug -Rs 27 · 08 Aug -Rs 27
         hold value        -Rs 1,618
         cancel value      -Rs 2,736
         income shock inferred

18 Aug   CUSTOMER LEFT - natural attrition while held
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: CHURNED · net value -Rs 2,775 · 30 days to resolution
Carried 1 cycles at Rs 34 in carrying cost
Mandate lost · LTV forfeited Rs 2,736
legacy_fixed on the same customer: presented 21 Jul, 23 Jul, 26 Jul; cancelled, 3 notifications sent, net -Rs 2,756
══════════════════════════════════════════════════════════════════════
```

## 4. A HOLD: nothing this cycle is worth presenting on

No date in the remaining cycle clears the money equation, so the mandate is carried rather than burned.

`--customer CUST_0046 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0046 · Rs 689 · nach · 6 months · LTV Rs 8,268
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

06 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

06 Jul   DECISION: HOLD
         Payday posterior: day 17 (conf 0.16)
         Shock probability: 0.57
         Best candidate 22 Jul scored -Rs 234
           P(clear) 0.38 x Rs 689            =    +Rs 264
           processing                        =     -Rs 15
           bounce risk 0.62 x Rs 500         =    -Rs 308
           opt-out risk 0.021 x Rs 8,268     =    -Rs 175
         Runners-up: 23 Jul -Rs 234 · 24 Jul -Rs 239 · 21 Jul -Rs 241
         hold value        -Rs 1,006
         cancel value      -Rs 8,268
         best date scores -234, not worth presenting

--- cycle 2 ---

06 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Aug   DECISION: HOLD
         Payday posterior: day 16 (conf 0.18)
         Shock probability: 0.57
         Best candidate 22 Aug scored -Rs 232
           P(clear) 0.39 x Rs 689            =    +Rs 265
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 307
           opt-out risk 0.021 x Rs 8,268     =    -Rs 175
         Runners-up: 21 Aug -Rs 234 · 23 Aug -Rs 235 · 24 Aug -Rs 242
         hold value        -Rs 995
         cancel value      -Rs 8,268
         best date scores -232, not worth presenting

--- cycle 3 ---

06 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Sep   DECISION: HOLD
         Payday posterior: day 15 (conf 0.19)
         Shock probability: 0.57
         Best candidate 21 Sep scored -Rs 231
           P(clear) 0.39 x Rs 689            =    +Rs 267
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 306
           opt-out risk 0.021 x Rs 8,268     =    -Rs 177
         Runners-up: 22 Sep -Rs 232 · 23 Sep -Rs 236 · 20 Sep -Rs 238
         hold value        -Rs 983
         cancel value      -Rs 8,268
         best date scores -231, not worth presenting

--- cycle 4 ---

06 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Oct   DECISION: HOLD
         Payday posterior: day 15 (conf 0.21)
         Shock probability: 0.57
         Best candidate 21 Oct scored -Rs 230
           P(clear) 0.39 x Rs 689            =    +Rs 268
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 306
           opt-out risk 0.021 x Rs 8,268     =    -Rs 177
         Runners-up: 20 Oct -Rs 232 · 22 Oct -Rs 233 · 23 Oct -Rs 238
         hold value        -Rs 976
         cancel value      -Rs 8,268
         best date scores -230, not worth presenting

--- cycle 5 ---

06 Nov   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Nov   DECISION: HOLD
         Payday posterior: day 14 (conf 0.23)
         Shock probability: 0.57
         Best candidate 21 Nov scored -Rs 230
           P(clear) 0.39 x Rs 689            =    +Rs 268
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 306
           opt-out risk 0.021 x Rs 8,268     =    -Rs 177
         Runners-up: 20 Nov -Rs 230 · 22 Nov -Rs 233 · 19 Nov -Rs 238
         hold value        -Rs 967
         cancel value      -Rs 8,268
         best date scores -230, not worth presenting

--- cycle 6 ---

06 Dec   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Dec   DECISION: HOLD
         Payday posterior: day 14 (conf 0.26)
         Shock probability: 0.57
         Best candidate 20 Dec scored -Rs 228
           P(clear) 0.39 x Rs 689            =    +Rs 269
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 305
           opt-out risk 0.022 x Rs 8,268     =    -Rs 178
         Runners-up: 21 Dec -Rs 230 · 19 Dec -Rs 233 · 22 Dec -Rs 235
         hold value        -Rs 961
         cancel value      -Rs 8,268
         best date scores -228, not worth presenting

--- cycle 7 ---

06 Jan   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Jan   DECISION: HOLD
         Payday posterior: day 14 (conf 0.29)
         Shock probability: 0.57
         Best candidate 20 Jan scored -Rs 228
           P(clear) 0.39 x Rs 689            =    +Rs 270
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 304
           opt-out risk 0.022 x Rs 8,268     =    -Rs 178
         Runners-up: 19 Jan -Rs 230 · 21 Jan -Rs 230 · 22 Jan -Rs 235
         hold value        -Rs 958
         cancel value      -Rs 8,268
         best date scores -228, not worth presenting

--- cycle 8 ---

06 Feb   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Feb   DECISION: HOLD
         Payday posterior: day 14 (conf 0.32)
         Shock probability: 0.57
         Best candidate 20 Feb scored -Rs 228
           P(clear) 0.39 x Rs 689            =    +Rs 270
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 304
           opt-out risk 0.022 x Rs 8,268     =    -Rs 178
         Runners-up: 19 Feb -Rs 229 · 21 Feb -Rs 231 · 22 Feb -Rs 236
         hold value        -Rs 950
         cancel value      -Rs 8,268
         best date scores -228, not worth presenting

--- cycle 9 ---

06 Mar   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Mar   DECISION: HOLD
         Payday posterior: day 14 (conf 0.36)
         Shock probability: 0.57
         Best candidate 19 Mar scored -Rs 227
           P(clear) 0.39 x Rs 689            =    +Rs 271
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 303
           opt-out risk 0.022 x Rs 8,268     =    -Rs 180
         Runners-up: 20 Mar -Rs 228 · 21 Mar -Rs 232 · 18 Mar -Rs 232
         hold value        -Rs 945
         cancel value      -Rs 8,268
         best date scores -227, not worth presenting

--- cycle 10 ---

06 Apr   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Apr   DECISION: HOLD
         Payday posterior: day 14 (conf 0.38)
         Shock probability: 0.57
         Best candidate 19 Apr scored -Rs 227
           P(clear) 0.39 x Rs 689            =    +Rs 271
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 303
           opt-out risk 0.022 x Rs 8,268     =    -Rs 180
         Runners-up: 20 Apr -Rs 228 · 18 Apr -Rs 231 · 21 Apr -Rs 232
         hold value        -Rs 943
         cancel value      -Rs 8,268
         best date scores -227, not worth presenting

--- cycle 11 ---

06 May   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 May   DECISION: HOLD
         Payday posterior: day 14 (conf 0.41)
         Shock probability: 0.57
         Best candidate 19 May scored -Rs 227
           P(clear) 0.39 x Rs 689            =    +Rs 272
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 303
           opt-out risk 0.022 x Rs 8,268     =    -Rs 180
         Runners-up: 20 May -Rs 229 · 18 May -Rs 229 · 21 May -Rs 233
         hold value        -Rs 941
         cancel value      -Rs 8,268
         best date scores -227, not worth presenting

--- cycle 12 ---

06 Jun   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

06 Jun   DECISION: HOLD
         Payday posterior: day 14 (conf 0.43)
         Shock probability: 0.57
         Best candidate 19 Jun scored -Rs 227
           P(clear) 0.39 x Rs 689            =    +Rs 272
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 303
           opt-out risk 0.022 x Rs 8,268     =    -Rs 180
         Runners-up: 18 Jun -Rs 228 · 20 Jun -Rs 229 · 21 Jun -Rs 233
         hold value        -Rs 936
         cancel value      -Rs 8,268
         best date scores -227, not worth presenting

──────────────────────────────────────────────────────────────────────
OUTCOME: HELD · net value -Rs 7,420 · 364 days to resolution
Carried 12 cycles at Rs 1,240 in carrying cost
Mandate retained · LTV preserved Rs 8,268
legacy_fixed on the same customer: presented 08 Jul, 10 Jul, 13 Jul; cancelled, 3 notifications sent, net -Rs 10,328
══════════════════════════════════════════════════════════════════════
```

## 5. A correct CANCEL on a customer whose income stopped

The oracle confirms no date could ever have cleared. Giving up is the right answer, and the stopping rule found it.

`--customer CUST_0147 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0147 · Rs 257 · nach · 3 months · LTV Rs 3,084
Failure class: expired_or_reissued_card (return code 54)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

13 Jul   DEBIT FAILED - return code 54
         Attempts used 1/4 · notifications sent 0

13 Jul   DECISION: NUDGE
         Shock probability: 0.00
         BOUND BY: failure_class - stored credential is dead; only the customer can replace it
         expired_or_reissued_card: no retry can help, but the customer can - nudging them to update the credential

14 Jul   DECISION: CANCEL
         Shock probability: 0.00
         BOUND BY: failure_class - stored credential is dead; only the customer can replace it
         hold value        -Rs 6,506
         cancel value      -Rs 3,084
         expired_or_reissued_card: stored credential is dead; only the customer can replace it

──────────────────────────────────────────────────────────────────────
OUTCOME: CANCELLED · net value -Rs 3,599 · 1 days to resolution
Mandate lost · LTV forfeited Rs 3,084
legacy_fixed on the same customer: presented 15 Jul, 17 Jul, 20 Jul; cancelled, 3 notifications sent, net -Rs 5,144
══════════════════════════════════════════════════════════════════════
```

## 6. A CANCEL that was wrong

The oracle says this mandate was still recoverable. The stopping rule gave up on a live customer - the failure mode worth watching.

`--customer CUST_0302 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0302 · Rs 359 · nach · 10 months · LTV Rs 4,308
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

23 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

23 Jul   DECISION: CANCEL
         Payday posterior: day 1 (conf 0.76)
         Shock probability: 1.00
         Best candidate 22 Aug scored -Rs 586
           P(clear) 0.00 x Rs 359            =       Rs 0
           processing                        =     -Rs 15
           bounce risk 1.00 x Rs 500         =    -Rs 500
           opt-out risk 0.017 x Rs 4,308     =     -Rs 71
         Runners-up: 21 Aug -Rs 586 · 20 Aug -Rs 586 · 19 Aug -Rs 586
         hold value        -Rs 7,209
         cancel value      -Rs 4,308
         income shock inferred

──────────────────────────────────────────────────────────────────────
OUTCOME: CANCELLED · net value -Rs 4,823 · 0 days to resolution
Mandate lost · LTV forfeited Rs 4,308
legacy_fixed on the same customer: presented 25 Jul, 27 Jul; opted_out, 2 notifications sent, net -Rs 5,338
══════════════════════════════════════════════════════════════════════
```

## 7. An opt-out on the pre-debit notification

RBI Sec 6(c) in action: the notice offered a way out and the customer took it. The whole annual LTV goes with it.

`--customer CUST_0181 --seed 2`

```
══════════════════════════════════════════════════════════════════════
CUST_0181 · Rs 3,801 · upi_autopay · 1 months · LTV Rs 45,612
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

13 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

13 Jul   DECISION: WAIT  (unchanged for 24 days)
         Payday posterior: day 1 (conf 0.31)
         Shock probability: 0.05
         Best candidate 07 Aug scored +Rs 1,647
           P(clear) 0.77 x Rs 3,801          =  +Rs 2,935
           processing                        =      -Rs 5
           bounce risk 0.23 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 45,612    =  -Rs 1,284
         Runners-up: 08 Aug +Rs 1,619 · 05 Aug +Rs 1,610 · 09 Aug +Rs 1,578
         best date is not yet due for commitment

06 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.31)
         Shock probability: 0.05
         Best candidate 07 Aug scored +Rs 1,647
           P(clear) 0.77 x Rs 3,801          =  +Rs 2,935
           processing                        =      -Rs 5
           bounce risk 0.23 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 45,612    =  -Rs 1,284
         Runners-up: 08 Aug +Rs 1,619 · 09 Aug +Rs 1,578 · 10 Aug +Rs 1,523
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

07 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

07 Aug   DECISION: WAIT  (unchanged for 2 days)
         Payday posterior: day 1 (conf 0.17)
         Shock probability: 0.05
         Best candidate 10 Aug scored +Rs 474
           P(clear) 0.60 x Rs 3,801          =  +Rs 2,266
           processing                        =      -Rs 5
           bounce risk 0.40 x Rs 0           =       Rs 0
           opt-out risk 0.039 x Rs 45,612    =  -Rs 1,787
         Runners-up: 09 Aug +Rs 423 · 11 Aug +Rs 393 · 08 Aug +Rs 355
         best date is not yet due for commitment

09 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.17)
         Shock probability: 0.05
         Best candidate 10 Aug scored +Rs 474
           P(clear) 0.60 x Rs 3,801          =  +Rs 2,266
           processing                        =      -Rs 5
           bounce risk 0.40 x Rs 0           =       Rs 0
           opt-out risk 0.039 x Rs 45,612    =  -Rs 1,787
         Runners-up: 11 Aug +Rs 393 · 12 Aug +Rs 308
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

10 Aug   OPTED OUT on the pre-debit notification - mandate revoked
         Attempts used 2/4 · notifications sent 2

──────────────────────────────────────────────────────────────────────
OUTCOME: OPTED_OUT · net value -Rs 45,622 · 28 days to resolution
Mandate lost · LTV forfeited Rs 45,612
legacy_fixed on the same customer: presented 15 Jul; opted_out, 1 notifications sent, net -Rs 45,617
══════════════════════════════════════════════════════════════════════
```

## 8. A held mandate that drifted away

Holding is not free. This customer was carried for months and left of their own accord before the money ever arrived.

`--customer CUST_0020 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0020 · Rs 109 · upi_autopay · 3 months · LTV Rs 1,308
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

07 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

07 Jul   DECISION: WAIT  (unchanged for 16 days)
         Payday posterior: day 16 (conf 0.14)
         Shock probability: 0.21
         Best candidate 24 Jul scored +Rs 34
           P(clear) 0.66 x Rs 109            =     +Rs 72
           processing                        =      -Rs 5
           bounce risk 0.34 x Rs 0           =       Rs 0
           opt-out risk 0.025 x Rs 1,308     =     -Rs 33
         Runners-up: 23 Jul +Rs 34 · 25 Jul +Rs 34 · 26 Jul +Rs 33
         best date is not yet due for commitment

23 Jul   DECISION: RETRY
         Payday posterior: day 16 (conf 0.14)
         Shock probability: 0.21
         Best candidate 24 Jul scored +Rs 34
           P(clear) 0.66 x Rs 109            =     +Rs 72
           processing                        =      -Rs 5
           bounce risk 0.34 x Rs 0           =       Rs 0
           opt-out risk 0.025 x Rs 1,308     =     -Rs 33
         Runners-up: 25 Jul +Rs 34 · 26 Jul +Rs 33 · 27 Jul +Rs 31
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

24 Jul   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

24 Jul   DECISION: WAIT  (unchanged for 7 days)
         Payday posterior: day 15 (conf 0.14)
         Shock probability: 0.21
         Best candidate 01 Aug scored +Rs 10
           P(clear) 0.55 x Rs 109            =     +Rs 59
           processing                        =      -Rs 5
           bounce risk 0.45 x Rs 0           =       Rs 0
           opt-out risk 0.034 x Rs 1,308     =     -Rs 45
         Runners-up: 31 Jul +Rs 9 · 30 Jul +Rs 8 · 26 Jul +Rs 7
         best date is not yet due for commitment

31 Jul   DECISION: RETRY
         Payday posterior: day 15 (conf 0.14)
         Shock probability: 0.21
         Best candidate 01 Aug scored +Rs 10
           P(clear) 0.55 x Rs 109            =     +Rs 59
           processing                        =      -Rs 5
           bounce risk 0.45 x Rs 0           =       Rs 0
           opt-out risk 0.034 x Rs 1,308     =     -Rs 45
         Runners-up: 02 Aug +Rs 7 · 03 Aug +Rs 4 · 04 Aug +Rs 1
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

01 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 3/4 · notifications sent 2

01 Aug   DECISION: HOLD
         Payday posterior: day 15 (conf 0.23)
         Shock probability: 0.57
         Best candidate 02 Aug scored -Rs 42
           P(clear) 0.23 x Rs 109            =     +Rs 25
           processing                        =      -Rs 5
           bounce risk 0.77 x Rs 0           =       Rs 0
           opt-out risk 0.048 x Rs 1,308     =     -Rs 62
         Runners-up: 03 Aug -Rs 44 · 04 Aug -Rs 46 · 05 Aug -Rs 48
         hold value        -Rs 69
         cancel value      -Rs 1,308
         best date scores -42, not worth presenting

--- cycle 2 ---

07 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Aug   DECISION: WAIT  (unchanged for 13 days)
         Payday posterior: day 15 (conf 0.27)
         Shock probability: 0.57
         Best candidate 21 Aug scored +Rs 2
           P(clear) 0.37 x Rs 109            =     +Rs 41
           processing                        =      -Rs 5
           bounce risk 0.63 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 22 Aug +Rs 2 · 23 Aug +Rs 1 · 20 Aug +Rs 1
         best date is not yet due for commitment

20 Aug   DECISION: RETRY
         Payday posterior: day 15 (conf 0.27)
         Shock probability: 0.57
         Best candidate 21 Aug scored +Rs 2
           P(clear) 0.37 x Rs 109            =     +Rs 41
           processing                        =      -Rs 5
           bounce risk 0.63 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 22 Aug +Rs 2 · 23 Aug +Rs 1 · 24 Aug +Rs 1
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

21 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

21 Aug   DECISION: HOLD
         Payday posterior: day 15 (conf 0.21)
         Shock probability: 0.57
         Best candidate 26 Aug scored -Rs 16
           P(clear) 0.32 x Rs 109            =     +Rs 35
           processing                        =      -Rs 5
           bounce risk 0.68 x Rs 0           =       Rs 0
           opt-out risk 0.035 x Rs 1,308     =     -Rs 46
         Runners-up: 25 Aug -Rs 16 · 27 Aug -Rs 16 · 28 Aug -Rs 18
         hold value        -Rs 85
         cancel value      -Rs 1,308
         best date scores -16, not worth presenting

--- cycle 3 ---

07 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Sep   DECISION: HOLD
         Payday posterior: day 15 (conf 0.26)
         Shock probability: 0.57
         Best candidate 22 Sep scored -Rs 1
           P(clear) 0.35 x Rs 109            =     +Rs 38
           processing                        =      -Rs 5
           bounce risk 0.65 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 23 Sep -Rs 1 · 25 Sep -Rs 1 · 24 Sep -Rs 1
         hold value        -Rs 68
         cancel value      -Rs 1,308
         best date scores -1, not worth presenting

--- cycle 4 ---

07 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Oct   DECISION: WAIT  (unchanged for 14 days)
         Payday posterior: day 15 (conf 0.28)
         Shock probability: 0.57
         Best candidate 22 Oct scored +Rs 1
           P(clear) 0.36 x Rs 109            =     +Rs 39
           processing                        =      -Rs 5
           bounce risk 0.64 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 20 Oct +Rs 1 · 21 Oct Rs 0 · 19 Oct Rs 0
         best date is not yet due for commitment

21 Oct   DECISION: RETRY
         Payday posterior: day 15 (conf 0.28)
         Shock probability: 0.57
         Best candidate 22 Oct scored +Rs 1
           P(clear) 0.36 x Rs 109            =     +Rs 39
           processing                        =      -Rs 5
           bounce risk 0.64 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 23 Oct Rs 0 · 24 Oct Rs 0 · 25 Oct -Rs 1
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

22 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

22 Oct   DECISION: HOLD
         Payday posterior: day 15 (conf 0.21)
         Shock probability: 0.87
         Best candidate 25 Oct scored -Rs 40
           P(clear) 0.10 x Rs 109            =     +Rs 11
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.035 x Rs 1,308     =     -Rs 46
         Runners-up: 26 Oct -Rs 40 · 27 Oct -Rs 41 · 24 Oct -Rs 41
         hold value        -Rs 797
         cancel value      -Rs 1,308
         income shock inferred

--- cycle 5 ---

07 Nov   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Nov   DECISION: HOLD
         Payday posterior: day 10 (conf 0.24)
         Shock probability: 0.87
         Best candidate 23 Nov scored -Rs 27
           P(clear) 0.10 x Rs 109            =     +Rs 11
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.025 x Rs 1,308     =     -Rs 33
         Runners-up: 25 Nov -Rs 27 · 24 Nov -Rs 27 · 22 Nov -Rs 27
         hold value        -Rs 797
         cancel value      -Rs 1,308
         income shock inferred

--- cycle 6 ---

07 Dec   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Dec   DECISION: HOLD
         Payday posterior: day 10 (conf 0.28)
         Shock probability: 0.87
         Best candidate 21 Dec scored -Rs 27
           P(clear) 0.11 x Rs 109            =     +Rs 12
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 22 Dec -Rs 27 · 23 Dec -Rs 27 · 20 Dec -Rs 27
         hold value        -Rs 797
         cancel value      -Rs 1,308
         income shock inferred

--- cycle 7 ---

07 Jan   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

07 Jan   DECISION: HOLD
         Payday posterior: day 10 (conf 0.30)
         Shock probability: 0.87
         Best candidate 21 Jan scored -Rs 27
           P(clear) 0.11 x Rs 109            =     +Rs 12
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.026 x Rs 1,308     =     -Rs 34
         Runners-up: 22 Jan -Rs 27 · 20 Jan -Rs 27 · 23 Jan -Rs 27
         hold value        -Rs 797
         cancel value      -Rs 1,308
         income shock inferred

06 Feb   CUSTOMER LEFT - natural attrition while held
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: CHURNED · net value -Rs 1,477 · 214 days to resolution
Carried 7 cycles at Rs 114 in carrying cost
Mandate lost · LTV forfeited Rs 1,308
legacy_fixed on the same customer: presented 09 Jul, 11 Jul, 14 Jul; cancelled, 3 notifications sent, net -Rs 1,328
══════════════════════════════════════════════════════════════════════
```

## 9. A NACH mandate, where a failed retry costs the customer Rs 500

Same equation, different rail. The bounce term raises the bar a date has to clear before it is worth presenting on.

`--customer CUST_0018 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0018 · Rs 808 · nach · 16 months · LTV Rs 9,696
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

27 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

27 Jul   DECISION: WAIT  (unchanged for 5 days)
         Payday posterior: day 1 (conf 1.00)
         Shock probability: 0.05
         Best candidate 02 Aug scored +Rs 490
           P(clear) 0.89 x Rs 808            =    +Rs 717
           processing                        =     -Rs 15
           bounce risk 0.11 x Rs 500         =     -Rs 57
           opt-out risk 0.016 x Rs 9,696     =    -Rs 155
         Runners-up: 01 Aug +Rs 490 · 03 Aug +Rs 488 · 04 Aug +Rs 484
         best date is not yet due for commitment

01 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 1.00)
         Shock probability: 0.05
         Best candidate 02 Aug scored +Rs 490
           P(clear) 0.89 x Rs 808            =    +Rs 717
           processing                        =     -Rs 15
           bounce risk 0.11 x Rs 500         =     -Rs 57
           opt-out risk 0.016 x Rs 9,696     =    -Rs 155
         Runners-up: 03 Aug +Rs 488 · 04 Aug +Rs 484 · 05 Aug +Rs 478
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

02 Aug   PRESENTMENT - SUCCEEDED, Rs 808 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 278 · 6 days to resolution
Mandate retained · LTV preserved Rs 9,696
legacy_fixed on the same customer: presented 29 Jul, 31 Jul, 03 Aug; recovered, 3 notifications sent, net -Rs 752
══════════════════════════════════════════════════════════════════════
```

## 10. A long wait for a distant payday

The failure lands just after payday, so the next opportunity is weeks out. A fixed schedule cannot reach it at all.

`--customer CUST_0035 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0035 · Rs 749 · nach · 7 months · LTV Rs 8,988
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

21 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

21 Jul   DECISION: HOLD
         Payday posterior: day 1 (conf 0.40)
         Shock probability: 0.57
         Best candidate 07 Aug scored -Rs 212
           P(clear) 0.39 x Rs 749            =    +Rs 289
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 307
           opt-out risk 0.020 x Rs 8,988     =    -Rs 179
         Runners-up: 08 Aug -Rs 216 · 05 Aug -Rs 220 · 06 Aug -Rs 222
         hold value        -Rs 1,010
         cancel value      -Rs 8,988
         best date scores -212, not worth presenting

--- cycle 2 ---

21 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Aug   DECISION: HOLD
         Payday posterior: day 1 (conf 0.42)
         Shock probability: 0.57
         Best candidate 07 Sep scored -Rs 212
           P(clear) 0.39 x Rs 749            =    +Rs 289
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 307
           opt-out risk 0.020 x Rs 8,988     =    -Rs 179
         Runners-up: 05 Sep -Rs 212 · 06 Sep -Rs 215 · 08 Sep -Rs 218
         hold value        -Rs 992
         cancel value      -Rs 8,988
         best date scores -212, not worth presenting

--- cycle 3 ---

21 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Sep   DECISION: HOLD
         Payday posterior: day 1 (conf 0.44)
         Shock probability: 0.57
         Best candidate 05 Oct scored -Rs 209
           P(clear) 0.39 x Rs 749            =    +Rs 293
           processing                        =     -Rs 15
           bounce risk 0.61 x Rs 500         =    -Rs 304
           opt-out risk 0.020 x Rs 8,988     =    -Rs 184
         Runners-up: 06 Oct -Rs 212 · 07 Oct -Rs 213 · 08 Oct -Rs 219
         hold value        -Rs 978
         cancel value      -Rs 8,988
         best date scores -209, not worth presenting

--- cycle 4 ---

21 Oct   PRESENTMENT - SUCCEEDED, Rs 749 recovered
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value -Rs 1,148 · 92 days to resolution
Carried 3 cycles at Rs 337 in carrying cost
Mandate retained · LTV preserved Rs 8,988
legacy_fixed on the same customer: presented 23 Jul, 25 Jul, 28 Jul; cancelled, 3 notifications sent, net -Rs 11,048
══════════════════════════════════════════════════════════════════════
```
