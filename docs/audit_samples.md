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
| 1 | A clean recovery | RECOVERED | Rs -111 | Rs -1,141 |
| 2 | Where the sequencer earns its keep | RECOVERED | Rs 66 | Rs -9,212 |
| 3 | Where the fixed schedule beat the sequencer | CHURNED | Rs -127,125 | Rs -115,520 |
| 4 | A HOLD: nothing this cycle is worth presenting on | HELD | Rs -3,781 | Rs -24,692 |
| 5 | A correct CANCEL on a customer whose income stopped | CANCELLED | Rs -2,195 | Rs -2,195 |
| 6 | A CANCEL that was wrong | CANCELLED | Rs -3,251 | Rs -4,796 |
| 7 | An opt-out on the pre-debit notification | OPTED_OUT | Rs -6,503 | Rs -6,503 |
| 8 | A held mandate that drifted away | CHURNED | Rs -11,570 | Rs -11,420 |
| 9 | An authentication gate, and the one action that opens it | RECOVERED | Rs 15,994 | Rs -201,752 |
| 10 | A NACH mandate, where a failed retry costs the customer Rs 500 | RECOVERED | Rs -347 | Rs -347 |
| 11 | A long wait for a distant payday | RECOVERED | Rs 167 | Rs 167 |

## 1. A clean recovery

The ordinary case: infer the payday, price the dates, present once on the best one.

`--customer CUST_0012 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0012 · Rs 419 · nach · 1 months · LTV Rs 5,028
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

25 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

25 Jul   DECISION: WAIT  (unchanged for 13 days)
         Payday posterior: day 1 (conf 0.32)
         Shock probability: 0.05
         Best candidate 08 Aug scored +Rs 79
           P(clear) 0.81 x Rs 419            =    +Rs 339
           processing                        =     -Rs 15
           bounce risk 0.19 x Rs 500         =     -Rs 95
           opt-out risk 0.030 x Rs 5,028     =    -Rs 150
         Runners-up: 09 Aug +Rs 79 · 07 Aug +Rs 75 · 10 Aug +Rs 74
         best date is not yet due for commitment

07 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.32)
         Shock probability: 0.05
         Best candidate 08 Aug scored +Rs 79
           P(clear) 0.81 x Rs 419            =    +Rs 339
           processing                        =     -Rs 15
           bounce risk 0.19 x Rs 500         =     -Rs 95
           opt-out risk 0.030 x Rs 5,028     =    -Rs 150
         Runners-up: 09 Aug +Rs 79 · 10 Aug +Rs 74 · 11 Aug +Rs 55
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

08 Aug   PRESENTMENT - SUCCEEDED, Rs 419 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value -Rs 111 · 14 days to resolution
Mandate retained · LTV preserved Rs 5,028
legacy_fixed on the same customer: presented 27 Jul, 29 Jul, 01 Aug; recovered, 3 notifications sent, net -Rs 1,141
══════════════════════════════════════════════════════════════════════
```

## 2. Where the sequencer earns its keep

The fixed schedule spent its whole budget and lost the mandate; the sequencer waited for payday.

`--customer CUST_0047 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0047 · Rs 596 · nach · 5 months · LTV Rs 7,152
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

22 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

22 Jul   DECISION: WAIT  (unchanged for 17 days)
         Payday posterior: day 1 (conf 0.28)
         Shock probability: 0.21
         Best candidate 09 Aug scored +Rs 86
           P(clear) 0.69 x Rs 596            =    +Rs 413
           processing                        =     -Rs 15
           bounce risk 0.31 x Rs 500         =    -Rs 154
           opt-out risk 0.022 x Rs 7,152     =    -Rs 158
         Runners-up: 08 Aug +Rs 84 · 10 Aug +Rs 80 · 07 Aug +Rs 74
         best date is not yet due for commitment

08 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.28)
         Shock probability: 0.21
         Best candidate 09 Aug scored +Rs 86
           P(clear) 0.69 x Rs 596            =    +Rs 413
           processing                        =     -Rs 15
           bounce risk 0.31 x Rs 500         =    -Rs 154
           opt-out risk 0.022 x Rs 7,152     =    -Rs 158
         Runners-up: 10 Aug +Rs 80 · 11 Aug +Rs 64 · 12 Aug +Rs 44
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

09 Aug   PRESENTMENT - SUCCEEDED, Rs 596 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 66 · 18 days to resolution
Mandate retained · LTV preserved Rs 7,152
legacy_fixed on the same customer: presented 24 Jul, 26 Jul, 29 Jul; cancelled, 3 notifications sent, net -Rs 9,212
══════════════════════════════════════════════════════════════════════
```

## 3. Where the fixed schedule beat the sequencer

This is here on purpose. A sample where the agent only ever wins is curated, and a reviewer should assume the worst of it.

`--customer CUST_0025 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0025 · Rs 9,625 · upi_autopay · 1 months · LTV Rs 115,500
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

21 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

21 Jul   DECISION: WAIT  (unchanged for 16 days)
         Payday posterior: day 1 (conf 0.33)
         Shock probability: 0.05
         Best candidate 07 Aug scored +Rs 4,463
           P(clear) 0.81 x Rs 9,625          =  +Rs 7,829
           processing                        =      -Rs 5
           bounce risk 0.19 x Rs 0           =       Rs 0
           opt-out risk 0.029 x Rs 115,500   =  -Rs 3,361
         Runners-up: 08 Aug +Rs 4,462 · 09 Aug +Rs 4,414 · 10 Aug +Rs 4,318
         best date is not yet due for commitment

06 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.33)
         Shock probability: 0.05
         Best candidate 07 Aug scored +Rs 4,463
           P(clear) 0.81 x Rs 9,625          =  +Rs 7,829
           processing                        =      -Rs 5
           bounce risk 0.19 x Rs 0           =       Rs 0
           opt-out risk 0.029 x Rs 115,500   =  -Rs 3,361
         Runners-up: 08 Aug +Rs 4,462 · 09 Aug +Rs 4,414 · 10 Aug +Rs 4,318
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

07 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

07 Aug   DECISION: WAIT  (unchanged for 2 days)
         Payday posterior: day 1 (conf 0.24)
         Shock probability: 0.05
         Best candidate 10 Aug scored +Rs 2,845
           P(clear) 0.78 x Rs 9,625          =  +Rs 7,476
           processing                        =      -Rs 5
           bounce risk 0.22 x Rs 0           =       Rs 0
           opt-out risk 0.040 x Rs 115,500   =  -Rs 4,626
         Runners-up: 11 Aug +Rs 2,694 · 12 Aug +Rs 2,503 · 09 Aug +Rs 2,479
         best date is not yet due for commitment

09 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.24)
         Shock probability: 0.05
         Best candidate 10 Aug scored +Rs 2,845
           P(clear) 0.78 x Rs 9,625          =  +Rs 7,476
           processing                        =      -Rs 5
           bounce risk 0.22 x Rs 0           =       Rs 0
           opt-out risk 0.040 x Rs 115,500   =  -Rs 4,626
         Runners-up: 11 Aug +Rs 2,694 · 12 Aug +Rs 2,503 · 13 Aug +Rs 2,271
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

10 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 3/4 · notifications sent 2

10 Aug   DECISION: HOLD
         Payday posterior: day 1 (conf 0.21)
         Shock probability: 0.21
         Best candidate 11 Aug scored -Rs 1,299
           P(clear) 0.54 x Rs 9,625          =  +Rs 5,155
           processing                        =      -Rs 5
           bounce risk 0.46 x Rs 0           =       Rs 0
           opt-out risk 0.056 x Rs 115,500   =  -Rs 6,449
         Runners-up: 12 Aug -Rs 1,367 · 13 Aug -Rs 1,483 · 14 Aug -Rs 1,641
         hold value        +Rs 2,764
         cancel value      -Rs 115,500
         best date scores -1,299, not worth presenting

--- cycle 2 ---

21 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Aug   DECISION: WAIT  (unchanged for 12 days)
         Payday posterior: day 1 (conf 0.23)
         Shock probability: 0.21
         Best candidate 03 Sep scored +Rs 2,810
           P(clear) 0.65 x Rs 9,625          =  +Rs 6,301
           processing                        =      -Rs 5
           bounce risk 0.35 x Rs 0           =       Rs 0
           opt-out risk 0.030 x Rs 115,500   =  -Rs 3,486
         Runners-up: 05 Sep +Rs 2,754 · 04 Sep +Rs 2,738 · 02 Sep +Rs 2,696
         best date is not yet due for commitment

02 Sep   DECISION: RETRY
         Payday posterior: day 1 (conf 0.23)
         Shock probability: 0.21
         Best candidate 03 Sep scored +Rs 2,810
           P(clear) 0.65 x Rs 9,625          =  +Rs 6,301
           processing                        =      -Rs 5
           bounce risk 0.35 x Rs 0           =       Rs 0
           opt-out risk 0.030 x Rs 115,500   =  -Rs 3,486
         Runners-up: 05 Sep +Rs 2,754 · 04 Sep +Rs 2,738 · 06 Sep +Rs 2,636
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

03 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

03 Sep   DECISION: WAIT  (unchanged for 7 days)
         Payday posterior: day 30 (conf 0.10)
         Shock probability: 0.21
         Best candidate 11 Sep scored +Rs 511
           P(clear) 0.53 x Rs 9,625          =  +Rs 5,123
           processing                        =      -Rs 5
           bounce risk 0.47 x Rs 0           =       Rs 0
           opt-out risk 0.040 x Rs 115,500   =  -Rs 4,606
         Runners-up: 12 Sep +Rs 482 · 10 Sep +Rs 448 · 13 Sep +Rs 385
         best date is not yet due for commitment

10 Sep   DECISION: RETRY
         Payday posterior: day 30 (conf 0.10)
         Shock probability: 0.21
         Best candidate 11 Sep scored +Rs 511
           P(clear) 0.53 x Rs 9,625          =  +Rs 5,123
           processing                        =      -Rs 5
           bounce risk 0.47 x Rs 0           =       Rs 0
           opt-out risk 0.040 x Rs 115,500   =  -Rs 4,606
         Runners-up: 12 Sep +Rs 482 · 13 Sep +Rs 385 · 14 Sep +Rs 241
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

11 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 3/4 · notifications sent 2

11 Sep   DECISION: HOLD
         Payday posterior: day 22 (conf 0.21)
         Shock probability: 0.21
         Best candidate 13 Sep scored -Rs 2,719
           P(clear) 0.38 x Rs 9,625          =  +Rs 3,691
           processing                        =      -Rs 5
           bounce risk 0.62 x Rs 0           =       Rs 0
           opt-out risk 0.055 x Rs 115,500   =  -Rs 6,404
         Runners-up: 14 Sep -Rs 2,735 · 12 Sep -Rs 2,805 · 15 Sep -Rs 2,817
         hold value        +Rs 2,699
         cancel value      -Rs 115,500
         best date scores -2,719, not worth presenting

--- cycle 3 ---

21 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Sep   DECISION: WAIT  (unchanged for 9 days)
         Payday posterior: day 22 (conf 0.30)
         Shock probability: 0.21
         Best candidate 01 Oct scored +Rs 2,609
           P(clear) 0.65 x Rs 9,625          =  +Rs 6,254
           processing                        =      -Rs 5
           bounce risk 0.35 x Rs 0           =       Rs 0
           opt-out risk 0.032 x Rs 115,500   =  -Rs 3,641
         Runners-up: 02 Oct +Rs 2,564 · 03 Oct +Rs 2,464 · 04 Oct +Rs 2,316
         best date is not yet due for commitment

30 Sep   DECISION: RETRY
         Payday posterior: day 22 (conf 0.30)
         Shock probability: 0.21
         Best candidate 01 Oct scored +Rs 2,609
           P(clear) 0.65 x Rs 9,625          =  +Rs 6,254
           processing                        =      -Rs 5
           bounce risk 0.35 x Rs 0           =       Rs 0
           opt-out risk 0.032 x Rs 115,500   =  -Rs 3,641
         Runners-up: 02 Oct +Rs 2,564 · 03 Oct +Rs 2,464 · 04 Oct +Rs 2,316
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

01 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

01 Oct   DECISION: WAIT  (unchanged for 3 days)
         Payday posterior: day 22 (conf 0.26)
         Shock probability: 0.21
         Best candidate 05 Oct scored +Rs 140
           P(clear) 0.52 x Rs 9,625          =  +Rs 4,973
           processing                        =      -Rs 5
           bounce risk 0.48 x Rs 0           =       Rs 0
           opt-out risk 0.042 x Rs 115,500   =  -Rs 4,827
         Runners-up: 06 Oct -Rs 42 · 08 Oct -Rs 73 · 07 Oct -Rs 124
         best date is not yet due for commitment

04 Oct   DECISION: RETRY
         Payday posterior: day 22 (conf 0.26)
         Shock probability: 0.21
         Best candidate 05 Oct scored +Rs 140
           P(clear) 0.52 x Rs 9,625          =  +Rs 4,973
           processing                        =      -Rs 5
           bounce risk 0.48 x Rs 0           =       Rs 0
           opt-out risk 0.042 x Rs 115,500   =  -Rs 4,827
         Runners-up: 06 Oct -Rs 42 · 08 Oct -Rs 73 · 07 Oct -Rs 124
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

05 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 3/4 · notifications sent 2

05 Oct   DECISION: HOLD
         Payday posterior: day 22 (conf 0.23)
         Shock probability: 0.21
         Best candidate 14 Oct scored -Rs 1,692
           P(clear) 0.49 x Rs 9,625          =  +Rs 4,718
           processing                        =      -Rs 5
           bounce risk 0.51 x Rs 0           =       Rs 0
           opt-out risk 0.055 x Rs 115,500   =  -Rs 6,404
         Runners-up: 15 Oct -Rs 1,740 · 13 Oct -Rs 1,760 · 16 Oct -Rs 1,845
         hold value        +Rs 2,841
         cancel value      -Rs 115,500
         best date scores -1,692, not worth presenting

--- cycle 4 ---

21 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Oct   DECISION: WAIT  (unchanged for 5 days)
         Payday posterior: day 22 (conf 0.41)
         Shock probability: 0.21
         Best candidate 27 Oct scored +Rs 1,883
           P(clear) 0.61 x Rs 9,625          =  +Rs 5,869
           processing                        =      -Rs 5
           bounce risk 0.39 x Rs 0           =       Rs 0
           opt-out risk 0.035 x Rs 115,500   =  -Rs 3,981
         Runners-up: 26 Oct +Rs 1,875 · 28 Oct +Rs 1,845 · 25 Oct +Rs 1,790
         best date is not yet due for commitment

26 Oct   DECISION: RETRY
         Payday posterior: day 22 (conf 0.41)
         Shock probability: 0.21
         Best candidate 27 Oct scored +Rs 1,883
           P(clear) 0.61 x Rs 9,625          =  +Rs 5,869
           processing                        =      -Rs 5
           bounce risk 0.39 x Rs 0           =       Rs 0
           opt-out risk 0.035 x Rs 115,500   =  -Rs 3,981
         Runners-up: 28 Oct +Rs 1,845 · 29 Oct +Rs 1,776 · 30 Oct +Rs 1,768
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

27 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

27 Oct   DECISION: HOLD
         Payday posterior: day 8 (conf 0.17)
         Shock probability: 0.57
         Best candidate 13 Nov scored -Rs 1,735
           P(clear) 0.30 x Rs 9,625          =  +Rs 2,845
           processing                        =      -Rs 5
           bounce risk 0.70 x Rs 0           =       Rs 0
           opt-out risk 0.040 x Rs 115,500   =  -Rs 4,574
         Runners-up: 12 Nov -Rs 1,755 · 14 Nov -Rs 1,785 · 15 Nov -Rs 1,865
         hold value        -Rs 7,559
         cancel value      -Rs 115,500
         best date scores -1,735, not worth presenting

--- cycle 5 ---

21 Nov   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Nov   DECISION: HOLD
         Payday posterior: day 22 (conf 0.30)
         Shock probability: 0.57
         Best candidate 08 Dec scored -Rs 538
           P(clear) 0.29 x Rs 9,625          =  +Rs 2,828
           processing                        =      -Rs 5
           bounce risk 0.71 x Rs 0           =       Rs 0
           opt-out risk 0.029 x Rs 115,500   =  -Rs 3,361
         Runners-up: 09 Dec -Rs 540 · 10 Dec -Rs 629 · 11 Dec -Rs 744
         hold value        -Rs 9,063
         cancel value      -Rs 115,500
         best date scores -538, not worth presenting

--- cycle 6 ---

21 Dec   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Dec   DECISION: HOLD
         Payday posterior: day 22 (conf 0.40)
         Shock probability: 0.57
         Best candidate 02 Jan scored -Rs 513
           P(clear) 0.31 x Rs 9,625          =  +Rs 3,023
           processing                        =      -Rs 5
           bounce risk 0.69 x Rs 0           =       Rs 0
           opt-out risk 0.031 x Rs 115,500   =  -Rs 3,531
         Runners-up: 03 Jan -Rs 531 · 05 Jan -Rs 602 · 04 Jan -Rs 607
         hold value        -Rs 6,812
         cancel value      -Rs 115,500
         best date scores -513, not worth presenting

--- cycle 7 ---

21 Jan   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Jan   DECISION: HOLD
         Payday posterior: day 22 (conf 0.47)
         Shock probability: 0.57
         Best candidate 02 Feb scored -Rs 300
           P(clear) 0.34 x Rs 9,625          =  +Rs 3,236
           processing                        =      -Rs 5
           bounce risk 0.66 x Rs 0           =       Rs 0
           opt-out risk 0.031 x Rs 115,500   =  -Rs 3,531
         Runners-up: 03 Feb -Rs 339 · 01 Feb -Rs 361 · 31 Jan -Rs 427
         hold value        -Rs 4,170
         cancel value      -Rs 115,500
         best date scores -300, not worth presenting

--- cycle 8 ---

21 Feb   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

21 Feb   DECISION: HOLD
         Payday posterior: day 22 (conf 0.51)
         Shock probability: 0.57
         Best candidate 03 Mar scored -Rs 122
           P(clear) 0.37 x Rs 9,625          =  +Rs 3,524
           processing                        =      -Rs 5
           bounce risk 0.63 x Rs 0           =       Rs 0
           opt-out risk 0.032 x Rs 115,500   =  -Rs 3,641
         Runners-up: 02 Mar -Rs 136 · 04 Mar -Rs 148 · 05 Mar -Rs 171
         hold value        -Rs 4,835
         cancel value      -Rs 115,500
         best date scores -122, not worth presenting

20 Mar   CUSTOMER LEFT - natural attrition while held
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: CHURNED · net value -Rs 127,125 · 242 days to resolution
Carried 8 cycles at Rs 11,550 in carrying cost
Mandate lost · LTV forfeited Rs 115,500
legacy_fixed on the same customer: presented 23 Jul, 25 Jul, 28 Jul; cancelled, 3 notifications sent, net -Rs 115,520
══════════════════════════════════════════════════════════════════════
```

## 4. A HOLD: nothing this cycle is worth presenting on

No date in the remaining cycle clears the money equation, so the mandate is carried rather than burned.

`--customer CUST_0065 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0065 · Rs 2,056 · upi_autopay · 2 months · LTV Rs 24,672
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

08 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

08 Jul   DECISION: WAIT  (unchanged for 15 days)
         Payday posterior: day 16 (conf 0.21)
         Shock probability: 0.21
         Best candidate 24 Jul scored +Rs 704
           P(clear) 0.67 x Rs 2,056          =  +Rs 1,381
           processing                        =      -Rs 5
           bounce risk 0.33 x Rs 0           =       Rs 0
           opt-out risk 0.027 x Rs 24,672    =    -Rs 672
         Runners-up: 23 Jul +Rs 698 · 25 Jul +Rs 695 · 26 Jul +Rs 675
         best date is not yet due for commitment

23 Jul   DECISION: RETRY
         Payday posterior: day 16 (conf 0.21)
         Shock probability: 0.21
         Best candidate 24 Jul scored +Rs 704
           P(clear) 0.67 x Rs 2,056          =  +Rs 1,381
           processing                        =      -Rs 5
           bounce risk 0.33 x Rs 0           =       Rs 0
           opt-out risk 0.027 x Rs 24,672    =    -Rs 672
         Runners-up: 25 Jul +Rs 695 · 26 Jul +Rs 675 · 27 Jul +Rs 645
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

24 Jul   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

24 Jul   DECISION: WAIT  (unchanged for 7 days)
         Payday posterior: day 16 (conf 0.19)
         Shock probability: 0.21
         Best candidate 01 Aug scored +Rs 210
           P(clear) 0.54 x Rs 2,056          =  +Rs 1,120
           processing                        =      -Rs 5
           bounce risk 0.46 x Rs 0           =       Rs 0
           opt-out risk 0.037 x Rs 24,672    =    -Rs 905
         Runners-up: 26 Jul +Rs 204 · 31 Jul +Rs 200 · 27 Jul +Rs 193
         best date is not yet due for commitment

31 Jul   DECISION: RETRY
         Payday posterior: day 16 (conf 0.19)
         Shock probability: 0.21
         Best candidate 01 Aug scored +Rs 210
           P(clear) 0.54 x Rs 2,056          =  +Rs 1,120
           processing                        =      -Rs 5
           bounce risk 0.46 x Rs 0           =       Rs 0
           opt-out risk 0.037 x Rs 24,672    =    -Rs 905
         Runners-up: 02 Aug +Rs 157 · 03 Aug +Rs 97 · 04 Aug +Rs 31
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 3 of 4 ✓
         highest positive expected value in the cycle

01 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 3/4 · notifications sent 2

01 Aug   DECISION: HOLD
         Payday posterior: day 16 (conf 0.28)
         Shock probability: 0.57
         Best candidate 02 Aug scored -Rs 781
           P(clear) 0.24 x Rs 2,056          =    +Rs 488
           processing                        =      -Rs 5
           bounce risk 0.76 x Rs 0           =       Rs 0
           opt-out risk 0.051 x Rs 24,672    =  -Rs 1,264
         Runners-up: 03 Aug -Rs 816 · 04 Aug -Rs 856 · 05 Aug -Rs 894
         hold value        -Rs 991
         cancel value      -Rs 24,672
         best date scores -781, not worth presenting

--- cycle 2 ---

08 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Aug   DECISION: WAIT  (unchanged for 13 days)
         Payday posterior: day 16 (conf 0.32)
         Shock probability: 0.57
         Best candidate 22 Aug scored +Rs 78
           P(clear) 0.37 x Rs 2,056          =    +Rs 767
           processing                        =      -Rs 5
           bounce risk 0.63 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 684
         Runners-up: 21 Aug +Rs 75 · 23 Aug +Rs 74 · 24 Aug +Rs 64
         best date is not yet due for commitment

21 Aug   DECISION: RETRY
         Payday posterior: day 16 (conf 0.32)
         Shock probability: 0.57
         Best candidate 22 Aug scored +Rs 78
           P(clear) 0.37 x Rs 2,056          =    +Rs 767
           processing                        =      -Rs 5
           bounce risk 0.63 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 684
         Runners-up: 23 Aug +Rs 74 · 24 Aug +Rs 64 · 25 Aug +Rs 58
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

22 Aug   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

22 Aug   DECISION: HOLD
         Payday posterior: day 16 (conf 0.24)
         Shock probability: 0.57
         Best candidate 26 Aug scored -Rs 269
           P(clear) 0.32 x Rs 2,056          =    +Rs 663
           processing                        =      -Rs 5
           bounce risk 0.68 x Rs 0           =       Rs 0
           opt-out risk 0.038 x Rs 24,672    =    -Rs 927
         Runners-up: 25 Aug -Rs 274 · 27 Aug -Rs 282 · 28 Aug -Rs 303
         hold value        -Rs 1,348
         cancel value      -Rs 24,672
         best date scores -269, not worth presenting

--- cycle 3 ---

08 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Sep   DECISION: WAIT  (unchanged for 12 days)
         Payday posterior: day 16 (conf 0.28)
         Shock probability: 0.57
         Best candidate 21 Sep scored +Rs 28
           P(clear) 0.35 x Rs 2,056          =    +Rs 724
           processing                        =      -Rs 5
           bounce risk 0.65 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 692
         Runners-up: 25 Sep +Rs 26 · 23 Sep +Rs 26 · 20 Sep +Rs 26
         best date is not yet due for commitment

20 Sep   DECISION: RETRY
         Payday posterior: day 16 (conf 0.28)
         Shock probability: 0.57
         Best candidate 21 Sep scored +Rs 28
           P(clear) 0.35 x Rs 2,056          =    +Rs 724
           processing                        =      -Rs 5
           bounce risk 0.65 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 692
         Runners-up: 25 Sep +Rs 26 · 23 Sep +Rs 26 · 22 Sep +Rs 23
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

21 Sep   PRESENTMENT - FAILED, return code 04
         Attempts used 2/4 · notifications sent 1

21 Sep   DECISION: HOLD
         Payday posterior: day 25 (conf 0.17)
         Shock probability: 0.87
         Best candidate 27 Sep scored -Rs 726
           P(clear) 0.10 x Rs 2,056          =    +Rs 201
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.037 x Rs 24,672    =    -Rs 922
         Runners-up: 28 Sep -Rs 728 · 26 Sep -Rs 728 · 30 Sep -Rs 730
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 4 ---

08 Oct   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Oct   DECISION: HOLD
         Payday posterior: day 16 (conf 0.19)
         Shock probability: 0.87
         Best candidate 26 Oct scored -Rs 467
           P(clear) 0.10 x Rs 2,056          =    +Rs 201
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.027 x Rs 24,672    =    -Rs 662
         Runners-up: 25 Oct -Rs 468 · 27 Oct -Rs 471 · 28 Oct -Rs 477
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 5 ---

08 Nov   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Nov   DECISION: HOLD
         Payday posterior: day 16 (conf 0.20)
         Shock probability: 0.87
         Best candidate 25 Nov scored -Rs 470
           P(clear) 0.10 x Rs 2,056          =    +Rs 202
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.027 x Rs 24,672    =    -Rs 667
         Runners-up: 26 Nov -Rs 473 · 24 Nov -Rs 475 · 23 Nov -Rs 476
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 6 ---

08 Dec   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Dec   DECISION: HOLD
         Payday posterior: day 9 (conf 0.20)
         Shock probability: 0.87
         Best candidate 23 Dec scored -Rs 471
           P(clear) 0.10 x Rs 2,056          =    +Rs 212
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 677
         Runners-up: 24 Dec -Rs 472 · 25 Dec -Rs 473 · 22 Dec -Rs 474
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 7 ---

08 Jan   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Jan   DECISION: HOLD
         Payday posterior: day 9 (conf 0.22)
         Shock probability: 0.87
         Best candidate 23 Jan scored -Rs 470
           P(clear) 0.10 x Rs 2,056          =    +Rs 212
           processing                        =      -Rs 5
           bounce risk 0.90 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 677
         Runners-up: 22 Jan -Rs 471 · 24 Jan -Rs 473 · 21 Jan -Rs 473
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 8 ---

08 Feb   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Feb   DECISION: HOLD
         Payday posterior: day 9 (conf 0.24)
         Shock probability: 0.87
         Best candidate 22 Feb scored -Rs 471
           P(clear) 0.11 x Rs 2,056          =    +Rs 218
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 684
         Runners-up: 23 Feb -Rs 471 · 21 Feb -Rs 472 · 24 Feb -Rs 474
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 9 ---

08 Mar   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Mar   DECISION: HOLD
         Payday posterior: day 9 (conf 0.28)
         Shock probability: 0.87
         Best candidate 22 Mar scored -Rs 472
           P(clear) 0.11 x Rs 2,056          =    +Rs 217
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 684
         Runners-up: 21 Mar -Rs 472 · 23 Mar -Rs 473 · 20 Mar -Rs 475
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 10 ---

08 Apr   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Apr   DECISION: HOLD
         Payday posterior: day 9 (conf 0.29)
         Shock probability: 0.87
         Best candidate 22 Apr scored -Rs 473
           P(clear) 0.11 x Rs 2,056          =    +Rs 216
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 684
         Runners-up: 21 Apr -Rs 473 · 23 Apr -Rs 475 · 20 Apr -Rs 475
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 11 ---

08 May   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 May   DECISION: HOLD
         Payday posterior: day 9 (conf 0.31)
         Shock probability: 0.87
         Best candidate 21 May scored -Rs 474
           P(clear) 0.11 x Rs 2,056          =    +Rs 222
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 692
         Runners-up: 22 May -Rs 474 · 20 May -Rs 476 · 23 May -Rs 477
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

--- cycle 12 ---

08 Jun   PRESENTMENT - FAILED, return code 04
         Attempts used 1/4 · notifications sent 0

08 Jun   DECISION: HOLD
         Payday posterior: day 9 (conf 0.33)
         Shock probability: 0.87
         Best candidate 21 Jun scored -Rs 475
           P(clear) 0.11 x Rs 2,056          =    +Rs 222
           processing                        =      -Rs 5
           bounce risk 0.89 x Rs 0           =       Rs 0
           opt-out risk 0.028 x Rs 24,672    =    -Rs 692
         Runners-up: 22 Jun -Rs 475 · 20 Jun -Rs 477 · 23 Jun -Rs 478
         hold value        -Rs 14,221
         cancel value      -Rs 24,672
         income shock inferred

──────────────────────────────────────────────────────────────────────
OUTCOME: HELD · net value -Rs 3,781 · 364 days to resolution
Carried 12 cycles at Rs 3,701 in carrying cost
Mandate retained · LTV preserved Rs 24,672
legacy_fixed on the same customer: presented 10 Jul, 12 Jul, 15 Jul; cancelled, 3 notifications sent, net -Rs 24,692
══════════════════════════════════════════════════════════════════════
```

## 5. A correct CANCEL on a customer whose income stopped

The oracle confirms no date could ever have cleared. Giving up is the right answer, and the stopping rule found it.

`--customer CUST_0093 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0093 · Rs 140 · nach · 4 months · LTV Rs 1,680
Failure class: customer_opted_out (return code 22)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

21 Jul   DEBIT FAILED - return code 22
         Attempts used 1/4 · notifications sent 0

21 Jul   DECISION: CANCEL
         Shock probability: 0.00
         BOUND BY: failure_class - consent withdrawn under RBI Sec 6(c); the mandate is revoked
         customer_opted_out: consent withdrawn under RBI Sec 6(c); the mandate is revoked

──────────────────────────────────────────────────────────────────────
OUTCOME: CANCELLED · net value -Rs 2,195 · 0 days to resolution
Mandate lost · LTV forfeited Rs 1,680
legacy_fixed on the same customer: presented no retries; cancelled, 0 notifications sent, net -Rs 2,195
══════════════════════════════════════════════════════════════════════
```

## 6. A CANCEL that was wrong

The oracle says this mandate was still recoverable. The stopping rule gave up on a live customer - the failure mode worth watching.

`--customer CUST_0233 --seed 2`

```
══════════════════════════════════════════════════════════════════════
CUST_0233 · Rs 228 · nach · 13 months · LTV Rs 2,736
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

22 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

22 Jul   DECISION: CANCEL
         Payday posterior: day 30 (conf 0.52)
         Shock probability: 1.00
         Best candidate 21 Aug scored -Rs 555
           P(clear) 0.00 x Rs 228            =       Rs 0
           processing                        =     -Rs 15
           bounce risk 1.00 x Rs 500         =    -Rs 500
           opt-out risk 0.015 x Rs 2,736     =     -Rs 40
         Runners-up: 20 Aug -Rs 555 · 19 Aug -Rs 555 · 18 Aug -Rs 555
         hold value        -Rs 6,306
         cancel value      -Rs 2,736
         income shock inferred

──────────────────────────────────────────────────────────────────────
OUTCOME: CANCELLED · net value -Rs 3,251 · 0 days to resolution
Mandate lost · LTV forfeited Rs 2,736
legacy_fixed on the same customer: presented 24 Jul, 26 Jul, 29 Jul; cancelled, 3 notifications sent, net -Rs 4,796
══════════════════════════════════════════════════════════════════════
```

## 7. An opt-out on the pre-debit notification

RBI Sec 6(c) in action: the notice offered a way out and the customer took it. The whole annual LTV goes with it.

`--customer CUST_0290 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0290 · Rs 499 · nach · 5 months · LTV Rs 5,988
Failure class: insufficient_funds (return code 04)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

24 Jul   DEBIT FAILED - return code 04
         Attempts used 1/4 · notifications sent 0

24 Jul   DECISION: WAIT  (unchanged for 8 days)
         Payday posterior: day 1 (conf 0.99)
         Shock probability: 0.05
         Best candidate 02 Aug scored +Rs 222
           P(clear) 0.89 x Rs 499            =    +Rs 442
           processing                        =     -Rs 15
           bounce risk 0.11 x Rs 500         =     -Rs 57
           opt-out risk 0.025 x Rs 5,988     =    -Rs 147
         Runners-up: 01 Aug +Rs 222 · 03 Aug +Rs 221 · 04 Aug +Rs 218
         best date is not yet due for commitment

01 Aug   DECISION: RETRY
         Payday posterior: day 1 (conf 0.99)
         Shock probability: 0.05
         Best candidate 02 Aug scored +Rs 222
           P(clear) 0.89 x Rs 499            =    +Rs 442
           processing                        =     -Rs 15
           bounce risk 0.11 x Rs 500         =     -Rs 57
           opt-out risk 0.025 x Rs 5,988     =    -Rs 147
         Runners-up: 03 Aug +Rs 221 · 04 Aug +Rs 218 · 05 Aug +Rs 214
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

02 Aug   OPTED OUT on the pre-debit notification - mandate revoked
         Attempts used 1/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: OPTED_OUT · net value -Rs 6,503 · 9 days to resolution
Mandate lost · LTV forfeited Rs 5,988
legacy_fixed on the same customer: presented 26 Jul; opted_out, 1 notifications sent, net -Rs 6,503
══════════════════════════════════════════════════════════════════════
```

## 8. A held mandate that drifted away

Holding is not free. This customer was carried for months and left of their own accord before the money ever arrived.

`--customer CUST_0096 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0096 · Rs 950 · upi_autopay · 2 months · LTV Rs 11,400
Failure class: expired_or_reissued_card (return code 54)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

20 Jul   DEBIT FAILED - return code 54
         Attempts used 1/4 · notifications sent 0

20 Jul   DECISION: NUDGE
         Shock probability: 0.00
         BOUND BY: failure_class - stored credential is dead; only the customer can replace it
         expired_or_reissued_card: no retry can help, but the customer can - nudging them to update the credential

21 Jul   DECISION: HOLD
         Shock probability: 0.00
         BOUND BY: failure_class - stored credential is dead; only the customer can replace it
         hold value        -Rs 6,596
         cancel value      -Rs 11,400
         expired_or_reissued_card: carrying the mandate while the customer fixes it beats writing off the LTV

19 Aug   CUSTOMER LEFT - natural attrition while held
         Attempts used 1/4 · notifications sent 0

──────────────────────────────────────────────────────────────────────
OUTCOME: CHURNED · net value -Rs 11,570 · 30 days to resolution
Carried 1 cycles at Rs 142 in carrying cost
Mandate lost · LTV forfeited Rs 11,400
legacy_fixed on the same customer: presented 22 Jul, 24 Jul, 27 Jul; cancelled, 3 notifications sent, net -Rs 11,420
══════════════════════════════════════════════════════════════════════
```

## 9. An authentication gate, and the one action that opens it

The only class where the agent can CHANGE the success probability rather than pick the best point on a fixed curve. Watch P(clear) rise as the nudges land, then the presentment go in behind them - and note that no notification was sent to do it, because a nudge is out of band and carries no Sec 6(c) opt-out.

`--customer CUST_0179 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0179 · Rs 16,811 · upi_autopay · 11 months · LTV Rs 201,732
Failure class: afa_incomplete (return code 1A)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

27 Jul   DEBIT FAILED - return code 1A
         Attempts used 1/4 · notifications sent 0

27 Jul   DECISION: NUDGE
         Payday posterior: day 27 (conf 0.15)
         Shock probability: 0.05
         Best candidate 03 Aug scored -Rs 410
           P(clear) 0.20 x Rs 16,811         =  +Rs 3,423
           processing                        =      -Rs 5
           bounce risk 0.80 x Rs 0           =       Rs 0
           opt-out risk 0.019 x Rs 201,732   =  -Rs 3,828
         Runners-up: 02 Aug -Rs 414 · 26 Aug -Rs 438 · 04 Aug -Rs 439
         authentication is the binding gate: asking is worth Rs 680 more than presenting into it

28 Jul   DECISION: NUDGE
         Payday posterior: day 27 (conf 0.15)
         Shock probability: 0.05
         Best candidate 02 Aug scored +Rs 674
           P(clear) 0.27 x Rs 16,811         =  +Rs 4,610
           processing                        =      -Rs 5
           bounce risk 0.73 x Rs 0           =       Rs 0
           opt-out risk 0.019 x Rs 201,732   =  -Rs 3,930
         Runners-up: 01 Aug +Rs 667 · 03 Aug +Rs 648 · 04 Aug +Rs 583
         authentication is the binding gate: asking is worth Rs 571 more than presenting into it

29 Jul   DECISION: WAIT  (unchanged for 2 days)
         Payday posterior: day 27 (conf 0.15)
         Shock probability: 0.05
         Best candidate 01 Aug scored +Rs 1,649
           P(clear) 0.34 x Rs 16,811         =  +Rs 5,703
           processing                        =      -Rs 5
           bounce risk 0.66 x Rs 0           =       Rs 0
           opt-out risk 0.020 x Rs 201,732   =  -Rs 4,049
         Runners-up: 02 Aug +Rs 1,634 · 03 Aug +Rs 1,581 · 31 Jul +Rs 1,537
         best date is not yet due for commitment

31 Jul   DECISION: RETRY
         Payday posterior: day 27 (conf 0.15)
         Shock probability: 0.05
         Best candidate 01 Aug scored +Rs 1,649
           P(clear) 0.34 x Rs 16,811         =  +Rs 5,703
           processing                        =      -Rs 5
           bounce risk 0.66 x Rs 0           =       Rs 0
           opt-out risk 0.020 x Rs 201,732   =  -Rs 4,049
         Runners-up: 02 Aug +Rs 1,634 · 03 Aug +Rs 1,581 · 04 Aug +Rs 1,483
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

01 Aug   PRESENTMENT - SUCCEEDED, Rs 16,811 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 15,994 · 5 days to resolution
Mandate retained · LTV preserved Rs 201,732
legacy_fixed on the same customer: presented 29 Jul, 31 Jul, 03 Aug; cancelled, 3 notifications sent, net -Rs 201,752
══════════════════════════════════════════════════════════════════════
```

## 10. A NACH mandate, where a failed retry costs the customer Rs 500

Same equation, different rail. The bounce term raises the bar a date has to clear before it is worth presenting on.

`--customer CUST_0032 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0032 · Rs 183 · nach · 5 months · LTV Rs 2,196
Failure class: bank_or_psp_downtime (return code 91)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

03 Jul   DEBIT FAILED - return code 91
         Attempts used 1/4 · notifications sent 0

03 Jul   DECISION: RETRY
         Payday posterior: day 1 (conf 0.41)
         Shock probability: 0.05
         Bank AXIS: P(outage) 0.015 · expected recovery 3.0h
         Best candidate 04 Jul scored +Rs 2
           P(clear) 0.86 x Rs 183            =    +Rs 157
           processing                        =     -Rs 15
           bounce risk 0.14 x Rs 500         =     -Rs 70
           opt-out risk 0.032 x Rs 2,196     =     -Rs 70
         Runners-up: 05 Jul Rs 0 · 06 Jul -Rs 3 · 07 Jul -Rs 9
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

04 Jul   PRESENTMENT - SUCCEEDED, Rs 183 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value -Rs 347 · 1 days to resolution
Mandate retained · LTV preserved Rs 2,196
legacy_fixed on the same customer: presented 05 Jul; recovered, 1 notifications sent, net -Rs 347
══════════════════════════════════════════════════════════════════════
```

## 11. A long wait for a distant payday

The failure lands just after payday, so the next opportunity is weeks out. A fixed schedule cannot reach it at all.

`--customer CUST_0028 --seed 1`

```
══════════════════════════════════════════════════════════════════════
CUST_0028 · Rs 177 · upi_autopay · 7 months · LTV Rs 2,124
Failure class: bank_or_psp_downtime (return code 91)
Policy: sequencer
══════════════════════════════════════════════════════════════════════

23 Jul   DEBIT FAILED - return code 91
         Attempts used 1/4 · notifications sent 0

23 Jul   DECISION: WAIT  (unchanged for 29 days)
         Payday posterior: day 10 (conf 0.12)
         Shock probability: 0.05
         Bank IDBI: P(outage) 0.015 · expected recovery 3.0h
         Best candidate 22 Aug scored +Rs 79
           P(clear) 0.70 x Rs 177            =    +Rs 125
           processing                        =      -Rs 5
           bounce risk 0.30 x Rs 0           =       Rs 0
           opt-out risk 0.019 x Rs 2,124     =     -Rs 41
         Runners-up: 21 Aug +Rs 73 · 20 Aug +Rs 68 · 19 Aug +Rs 62
         best date is not yet due for commitment

21 Aug   DECISION: RETRY
         Payday posterior: day 10 (conf 0.12)
         Shock probability: 0.05
         Bank IDBI: P(outage) 0.015 · expected recovery 3.0h
         Best candidate 22 Aug scored +Rs 79
           P(clear) 0.70 x Rs 177            =    +Rs 125
           processing                        =      -Rs 5
           bounce risk 0.30 x Rs 0           =       Rs 0
           opt-out risk 0.019 x Rs 2,124     =     -Rs 41
         Compliance: committed 26h ahead (need 24h) ✓
                     02:00 inside 00:00-07:00 window ✓
                     attempt 2 of 4 ✓
         highest positive expected value in the cycle

22 Aug   PRESENTMENT - SUCCEEDED, Rs 177 recovered
         Attempts used 2/4 · notifications sent 1

──────────────────────────────────────────────────────────────────────
OUTCOME: RECOVERED · net value +Rs 167 · 30 days to resolution
Mandate retained · LTV preserved Rs 2,124
legacy_fixed on the same customer: presented 25 Jul; recovered, 1 notifications sent, net +Rs 167
══════════════════════════════════════════════════════════════════════
```
