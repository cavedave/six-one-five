# Plan for a 10-Hour `(6,1,7)` Search

## Objective

Search for integer solutions to

\[
a_1^6+a_2^6+a_3^6+a_4^6+a_5^6+a_6^6+a_7^6=B^6
\]

with

\[
0<a_1\le a_2\le a_3\le a_4\le a_5\le a_6\le a_7<B.
\]

The immediate goals are:

1. Recover the known solution

   \[
   74^6+234^6+402^6+474^6+702^6+894^6+1077^6=1141^6.
   \]

2. Confirm that the existing sixth-power search architecture works for seven left-side terms.

3. Use a 10-hour overnight run to search the largest reliable contiguous range possible.

4. Simultaneously check the six-term lower-order case

   \[
   a_1^6+\cdots+a_6^6=B^6
   \]

   whenever this can be done with little extra work.

5. Save enough timing and rejection statistics to estimate future 24-hour and one-week searches.

The search must prioritize correctness and reproducibility over merely reporting a large upper bound.

---

# Mathematical filters

## Primitive solutions

Only primitive solutions should be reported.

Require:

\[
\gcd(a_1,a_2,a_3,a_4,a_5,a_6,a_7,B)=1.
\]

Nonprimitive solutions are scaled copies of smaller solutions and are not interesting as new results.

---

## Parity filter: modulo 8

For every integer \(x\),

\[
x^6\equiv
\begin{cases}
0\pmod 8,&x\text{ even},\\
1\pmod 8,&x\text{ odd}.
\end{cases}
\]

Therefore:

\[
a_1^6+\cdots+a_7^6\equiv
\#\{\text{odd }a_i\}
\pmod 8.
\]

For a primitive solution:

- \(B\) cannot be even, because then all seven \(a_i\) would have to be even.
- Therefore \(B\) is odd.
- Exactly one of the seven \(a_i\) is odd.

Required condition:

```text
B is odd
exactly one ai is odd
```

This should be enforced incrementally during tuple construction.

---

## Modulo 9 filter

For every integer \(x\),

\[
x^6\equiv
\begin{cases}
0\pmod 9,&3\mid x,\\
1\pmod 9,&3\nmid x.
\end{cases}
\]

For a primitive solution:

- \(3\nmid B\).
- Exactly one of the seven \(a_i\) is not divisible by 3.

Required condition:

```text
B mod 3 != 0
exactly one ai is not divisible by 3
```

Again, enforce this incrementally.

---

## Modulo 7 filter

For every integer \(x\),

\[
x^6\equiv
\begin{cases}
0\pmod 7,&7\mid x,\\
1\pmod 7,&7\nmid x.
\end{cases}
\]

Let \(r\) be the number of left-side terms not divisible by 7.

There are two primitive branches.

### Branch A: \(7\nmid B\)

Then

\[
r\equiv1\pmod7.
\]

Since \(0\le r\le7\),

\[
r=1.
\]

Required condition:

```text
B mod 7 != 0
exactly one ai is not divisible by 7
```

This is the branch most similar to the old `(6,1,5)` search.

### Branch B: \(7\mid B\)

Then

\[
r\equiv0\pmod7.
\]

The possibilities are \(r=0\) or \(r=7\).

- \(r=0\) would make all terms divisible by 7 and produce a nonprimitive solution.
- Therefore the primitive case requires \(r=7\).

Required condition:

```text
B mod 7 == 0
none of the ai are divisible by 7
```

This branch is essential.

The known \(B=1141\) solution is in Branch B because:

```text
1141 mod 7 == 0
all seven left-side bases are nonzero mod 7
```

Do not use the old rule:

```text
gcd(B,42) == 1
```

because it would reject the known solution.

---

## Modulo 60 square-sum filter

For every integer \(x\),

\[
x^6\equiv x^2\pmod{60}.
\]

Therefore every solution must satisfy:

\[
a_1^2+a_2^2+a_3^2+a_4^2+a_5^2+a_6^2+a_7^2
\equiv B^2\pmod{60}.
\]

This is a cheap secondary filter.

Precompute:

```cpp
square_mod_60[x] = (x * x) % 60;
```

Use the running square-residue sum during tuple construction.

---

## Combined small modulus

The least common multiple of 8, 9, and 7 is:

\[
504.
\]

Precompute for every base:

```cpp
struct BaseMetadata {
    uint64_t pow6_exact_or_residue;
    uint16_t pow6_mod_504;
    uint8_t square_mod_60;
    bool is_odd;
    bool nondivisible_by_3;
    bool nondivisible_by_7;
};
```

The exact type of `pow6_exact_or_residue` depends on the current search range.

For the validation search near 1141, native 128-bit exact values are sufficient.

---

# Search decomposition

## Core decomposition

Rewrite the equation as:

\[
a_1^6+a_2^6+a_3^6+a_4^6+a_5^6
=
B^6-a_6^6-a_7^6.
\]

The outer search chooses:

```text
B
a7
a6
```

and constructs:

\[
R=B^6-a_7^6-a_6^6.
\]

The inner solver searches for five ordered positive terms:

\[
a_1^6+a_2^6+a_3^6+a_4^6+a_5^6=R
\]

subject to:

\[
a_1\le a_2\le a_3\le a_4\le a_5\le a_6.
\]

The existing `(6,1,5)` machinery should be generalized so that the target is an arbitrary exact residual \(R\), not necessarily a perfect sixth power.

Suggested interface:

```cpp
solve_five_sum(
    target = R,
    max_term = a6,
    remaining_odd_count,
    remaining_nondiv3_count,
    remaining_nondiv7_count,
    required_square_sum_mod_60
);
```

---

# Incremental residue budgets

The full seven-term solution must contain:

```text
exactly one odd ai
exactly one ai not divisible by 3
Branch A: exactly one ai not divisible by 7
Branch B: exactly seven ai not divisible by 7
```

After selecting `a7` and `a6`, calculate the remaining quotas.

```cpp
remaining_odd =
    1
    - is_odd[a7]
    - is_odd[a6];

remaining_nondiv3 =
    1
    - nondiv3[a7]
    - nondiv3[a6];
```

For Branch A:

```cpp
remaining_nondiv7 =
    1
    - nondiv7[a7]
    - nondiv7[a6];
```

For Branch B:

```cpp
remaining_nondiv7 =
    7
    - nondiv7[a7]
    - nondiv7[a6];
```

Reject the outer pair immediately if any remaining quota is negative.

For Branch B, every selected term must be nonzero modulo 7, so this can be enforced directly without maintaining a general quota.

The five-term solver must enforce the exact remaining counts.

---

# Bounds for `a7` and `a6`

## Basic exact bounds

Because all terms are positive:

\[
a_7^6<B^6.
\]

Therefore:

```text
a7 < B
```

After choosing `a7`:

\[
a_6^6 < B^6-a_7^6.
\]

Therefore:

\[
a_6 \le
\left\lfloor
(B^6-a_7^6)^{1/6}
\right\rfloor.
\]

Also require:

```text
a6 <= a7
```

After choosing `a6`, reject when:

```text
R <= 0
```

---

## Five-term feasibility bounds

The remaining five terms are ordered and bounded by `a6`.

Necessary lower bound:

\[
5a_1^6\le R.
\]

Necessary upper bound:

\[
R\le5a_6^6.
\]

A simple branch rejection is:

```cpp
if (R > 5 * pow6[a6]) reject;
```

Within the five-term solver, after selecting a candidate `a5`, use:

\[
R-a_5^6
\]

and corresponding four-term upper and lower bounds.

The exact bounds should be checked before expensive probes.

---

# Heuristic search lanes

Do not use only balanced term bands.

The known solution has approximate term ratios:

```text
a1/B = 0.065
a2/B = 0.205
a3/B = 0.352
a4/B = 0.415
a5/B = 0.615
a6/B = 0.783
a7/B = 0.944
```

A balanced-only search would likely miss it.

Use at least three lanes.

## Lane 1: broad correctness lane

Purpose:

```text
recover the known B=1141 solution
avoid strong heuristic exclusions
establish an unbiased baseline
```

Characteristics:

```text
broad a7 range
broad a6 range
exact residue filters
exact monotonic bounds
minimal heuristic banding
```

This lane should run first.

## Lane 2: dominant-largest-term lane

Suggested initial ranges:

```text
0.88B <= a7 < B
0.55B <= a6 <= a7
```

These should be adjusted after observing the known solution and benchmark statistics.

## Lane 3: balanced partition lane

Purpose:

```text
search tuples where sixth-power contributions are more evenly distributed
```

This lane may use generated simplex-derived bands, but it must remain separate from the broad lane.

Do not claim the entire range has been searched if only heuristic lanes were used.

Distinguish:

```text
complete search
broad constrained search
heuristic search
```

in checkpoints and output.

---

# Lower-order checks

## Check `(6,1,6)`

When a six-term partial sum has already been computed:

\[
S_6=a_1^6+\cdots+a_6^6,
\]

probe whether:

\[
S_6=B^6.
\]

Equivalent zero-padding representation:

\[
0^6+a_1^6+\cdots+a_6^6=B^6.
\]

This check should be enabled if the loop structure exposes `S6` without reconstructing it expensively.

Any `(6,1,6)` hit is highly significant.

Required handling:

```text
remove the computational zero
verify with exact arithmetic
report reduced signature (6,1,6)
```

## Optional pair-target checks

If pair-target tables already exist, the same generated six- or seven-term sums may also be checked against:

\[
b_1^6+b_2^6.
\]

This can report:

```text
(6,2,6)
(6,2,7)
```

Only enable this if the additional lookup does not materially reduce throughput.

---

# Data structures

## Power table

For each base from 0 through `B_max`, store:

```cpp
struct PowerEntry {
    unsigned __int128 pow6;
    uint16_t mod504;
    uint8_t square_mod60;
    uint8_t flags;
};
```

Suggested flags:

```text
bit 0: odd
bit 1: nondivisible by 3
bit 2: nondivisible by 7
```

For ranges above the exact 128-bit limit, replace `pow6` with:

```text
low 128-bit residue for GPU filtering
plus exact wider value for CPU verification
```

This is not required for the initial validation range.

---

## Candidate record

A candidate must retain enough information for exact reconstruction.

```cpp
struct Candidate617 {
    uint32_t B;
    uint32_t a[7];
    uint8_t mod7_branch;
    uint8_t search_lane;
};
```

If the GPU only produces partial candidates, store the outer variables and the indexes necessary to recover the five-term tuple.

---

## Checkpoint record

Suggested JSON Lines format:

```json
{
  "kernel_version": "617-v1",
  "branch": "B-divisible-by-7",
  "lane": "broad",
  "B_start": 1100,
  "B_end": 1160,
  "completed": true,
  "runtime_seconds": 312.7,
  "outer_pairs": 1842931,
  "five_sum_calls": 43012,
  "mod_count_rejections": 1720000,
  "mod60_rejections": 8140,
  "fingerprint_hits": 4,
  "exact_checks": 4,
  "solutions": 1
}
```

Write one record after every completed block.

---

# Exact verification

Every reported candidate must be rechecked independently.

Pseudo-code:

```cpp
bool verify_617(const Candidate617& c) {
    BigInt left = 0;

    for (int i = 0; i < 7; ++i) {
        left += pow_exact(c.a[i], 6);
    }

    BigInt right = pow_exact(c.B, 6);

    if (left != right) {
        return false;
    }

    BigInt g = gcd_all(c.a[0], ..., c.a[6], c.B);

    if (g != 1) {
        return false;
    }

    return true;
}
```

At the initial ranges, exact unsigned 128-bit verification is sufficient.

Use a second independent verification implementation if possible.

For example:

```text
main program: unsigned __int128
verification script: Python integers
```

---

# Required validation case

Before any large run, the program must recover:

```text
B = 1141
left = 74, 234, 402, 474, 702, 894, 1077
```

The validation search should cover:

```text
B from 1100 through 1160
both mod-7 branches
broad lane
no aggressive ratio filters
```

Assertions for the known solution:

```text
B is odd
B is not divisible by 3
B is divisible by 7

exactly one ai is odd
exactly one ai is not divisible by 3
all seven ai are not divisible by 7
```

If the solution is not found:

```text
do not start the overnight run
```

Investigate:

```text
ordering assumptions
ratio bands
mod-7 branch handling
partial residue quotas
five-term residual interface
exact reconstruction
candidate output
```

---

# Ten-hour execution plan

## Phase 0: build and unit tests

Target duration:

```text
before the overnight window
```

Required tests:

```text
power table values agree with Python
mod-8 count rules
mod-9 count rules
both mod-7 branches
mod-60 square-sum rule
known B=1141 identity passes all filters
synthetic invalid candidates are rejected
checkpoint restart works
```

---

## Phase 1: known-solution recovery

Duration:

```text
up to 20 minutes
```

Range:

```text
B = 1100 to 1160
```

Configuration:

```text
broad lane
Branch A enabled
Branch B enabled
all exact modular filters enabled
no narrow heuristic term bands
```

Success condition:

```text
recover the B=1141 solution
```

Stop condition:

```text
if not recovered, stop and debug
```

---

## Phase 2: throughput calibration

Duration:

```text
40 minutes
```

Suggested calibration blocks:

```text
1200 to 1250
1500 to 1550
2000 to 2050
```

Use the same broad coverage for every block.

Record:

```text
B values per second
outer (a7,a6) pairs per second
five-term solver calls per second
five-term probes per second
rejection rates by filter
GPU utilization
GPU memory usage
CPU utilization
candidate survivors
exact checks
```

Fit a rough empirical scaling model:

\[
T(B)\approx cB^p.
\]

Do not assume a theoretical exponent without comparing it to measurements.

---

## Phase 3: choose the overnight range

Remaining time:

```text
approximately 9 hours
```

Use the benchmark to select a contiguous range that should complete within 8 hours.

Reserve approximately 1 hour for:

```text
unexpected slowdown
checkpoint flushing
final verification
rerunning the last incomplete block
```

Search order:

```text
lowest unsearched B first
ascending contiguous blocks
```

Suggested block width:

```text
small enough that one block finishes in 5 to 20 minutes
```

This makes restart and progress reporting reliable.

---

## Phase 4: branch allocation

Initial compute allocation:

```text
45 percent Branch A
45 percent Branch B
10 percent broad or skewed coverage
```

Do not allocate Branch B only according to the fraction of `B` values divisible by 7.

Branch B has different residue structure and contains the known smallest solution.

After approximately two hours, compare:

```text
throughput
survivor count
verified hit rate
GPU utilization
```

Rebalance only if one branch is clearly more productive or much cheaper.

---

## Phase 5: final verification

Final 30 to 60 minutes:

```text
rerun every candidate with independent exact arithmetic
deduplicate
check primitive gcd
sort left-side terms
classify reduced signature
write human-readable and machine-readable reports
```

Suggested result files:

```text
results/617_candidates.jsonl
results/617_verified.jsonl
results/617_solutions.txt
results/617_checkpoints.jsonl
results/617_metrics.csv
```

---

# Runtime and monitoring

## Recommended process launch

Example:

```bash
nohup env OMP_NUM_THREADS=20 nice -n 5 \
  ./solve_617 \
  --config configs/617_overnight.json \
  > runs/617_overnight.out \
  2> runs/617_overnight.log &

echo $! > runs/617_overnight.pid
disown
```

Use a less aggressive nice value than `12` if the machine is otherwise idle.

Suggested:

```text
nice -n 5
```

or:

```text
nice -n 0
```

if full priority is acceptable.

---

## Monitoring commands

Process:

```bash
ps -o user,pid,ppid,%cpu,%mem,etime,cmd \
  -p "$(cat runs/617_overnight.pid)"
```

GPU:

```bash
watch -n 2 nvidia-smi
```

Log:

```bash
tail -f runs/617_overnight.log
```

Completed blocks:

```bash
tail -20 results/617_checkpoints.jsonl
```

---

# Configuration example

Suggested machine-readable configuration:

```json
{
  "problem": {
    "power": 6,
    "right_terms": 1,
    "left_terms": 7
  },
  "search": {
    "B_start": 1161,
    "B_end": 5000,
    "block_width": 50,
    "ordered_terms": true,
    "primitive_only": true
  },
  "branches": {
    "B_not_divisible_by_7": true,
    "B_divisible_by_7": true
  },
  "filters": {
    "parity_count": true,
    "nondiv3_count": true,
    "nondiv7_count": true,
    "square_sum_mod_60": true,
    "residue_mod_504": true
  },
  "lanes": {
    "broad": true,
    "dominant_largest": true,
    "balanced": false
  },
  "lower_checks": {
    "check_616": true,
    "check_pair_targets": false
  },
  "output": {
    "checkpoint_every_block": true,
    "exact_verify": true
  }
}
```

The final `B_end` must be selected from benchmark results, not guessed in advance.

---

# LLM implementation instructions

An LLM modifying the existing `(6,1,5)` code should follow this order.

## Step 1

Locate the function that solves:

\[
a_1^6+\cdots+a_5^6=B^6.
\]

Refactor it so the right side is an arbitrary target residual:

```cpp
solve_five_sum(target, max_term, residue_budget);
```

Do not require the target to be a perfect sixth power.

## Step 2

Add an outer loop over:

```text
B
a7
a6
```

Construct:

```cpp
R = pow6[B] - pow6[a7] - pow6[a6];
```

Reject nonpositive residuals.

## Step 3

Add the exact mod-8, mod-9, and two-branch mod-7 count logic.

Do not reuse `gcd(B,42)==1`.

## Step 4

Pass remaining residue quotas into the five-term solver.

The five-term solver must reject partial tuples that exceed a remaining quota.

## Step 5

Add the mod-60 square-sum requirement.

For the five remaining terms, require:

```cpp
remaining_square_mod60 =
    (
        square_mod60[B]
        - square_mod60[a7]
        - square_mod60[a6]
    ) mod 60;
```

The five-term sum must match this residue.

## Step 6

Add exact candidate reconstruction and independent verification.

## Step 7

Add the known `B=1141` regression test.

## Step 8

Add checkpointing and metrics before attempting a long run.

## Step 9

Run calibration blocks and use measured throughput to select the overnight range.

---

# Common failure modes

## Incorrectly applying `gcd(B,42)==1`

This rejects the known solution.

Correct rule:

```text
gcd(B,6)==1
then split on B mod 7
```

## Applying five-term ratio bands to seven-term solutions

The known solution is highly skewed.

Use a broad lane for correctness.

## Checking residue counts only after generating all seven terms

This wastes most of the search.

Carry remaining quotas through every level.

## Treating Branch B as negligible

Branch B contains the known smallest solution.

Search it explicitly.

## Reporting a heuristic range as complete

Only claim complete coverage when every allowed ordered tuple in the stated range
has been searched after mathematically valid filters.

## Failing to checkpoint

A 10-hour run must be restartable at small block boundaries.

## Exact-verifying with the same buggy arithmetic path

Use an independent implementation, preferably Python integers or a separate
wide-integer verifier.

---

# Success criteria

The implementation is considered validated when all of the following are true:

- [ ] The known \(B=1141\) solution is recovered.
- [ ] Both mod-7 branches are exercised by tests.
- [ ] The search reports no false exact solutions.
- [ ] Checkpoint restart reproduces the same results.
- [ ] Throughput metrics are recorded for multiple \(B\) ranges.
- [ ] The overnight run completes a documented contiguous range.
- [ ] `(6,1,6)` partial checks are enabled or explicitly documented as disabled.
- [ ] Every candidate is independently exact-verified.
- [ ] Results distinguish complete coverage from heuristic coverage.

---

# Recommended first overnight objective

Do not choose an ambitious final upper bound before benchmarking.

The first overnight objective is:

> Recover the known \(B=1141\) identity, calibrate scaling at several larger
> values of \(B\), then complete the largest contiguous range that the measured
> implementation can reliably finish within the remaining nine hours while
> searching both primitive mod-7 branches and opportunistically checking
> `(6,1,6)` partial sums.

The most important correctness rule is:

> For primitive `(6,1,7)` solutions, \(B\) is odd and not divisible by 3.
> If \(7\nmid B\), exactly one left-side term is not divisible by 7.
> If \(7\mid B\), all seven left-side terms are not divisible by 7.
