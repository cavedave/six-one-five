# The (6,1,5) equation $a_1^6+a_2^6+a_3^6+a_4^6+a_5^6=B^6$

## A complete set of necessary conditions — congruences, factorization identities, descent relations, parametric obstructions — and what they do and do not prove

**Problem.** Determine whether the Lander–Parkin–Selfridge threshold equation

$$a_1^6+a_2^6+a_3^6+a_4^6+a_5^6=B^6,\qquad 1\le a_1<\dots<a_5<B,\qquad \gcd(a_1,\dots,a_5,B)=1$$

has solutions. This is the $(k,m,n)=(6,1,5)$ case of *equal sums of like powers*, sitting exactly at the Lander–Parkin–Selfridge boundary $k=m+n$ (so LPS does **not** forbid it), while a solution would refute Euler's sum-of-powers conjecture for $k=6$.[^1^][^2^]

**Status (as of the latest records).**

- No solution is known to any $(6,1,n)$ with $n<7$; the smallest $(6,1,7)$ is $1141^6=1077^6+894^6+702^6+474^6+402^6+234^6+74^6$ (Lander–Parkin–Selfridge, 1966).[^3^]
- The complementary signature $(6,2,5)$ *is* solvable (Resta–Meyrignac 2003).[^3^][^5^]
- Exhaustive computation has excluded all $k=6$ solutions with final term $\le 730{,}000$ (known since 2002).[^1^]
- Newton–Rouse (2021) proved $164634913=(44/5)^6+(117/5)^6$ is the smallest integer that is a sum of two *rational* sixth powers but not of two *integer* sixth powers — the closest proved non-existence result on a Fermat-type sixth-power variety (Section 8).[^4^]

**What this note is.** A complete, finite checklist of necessary conditions (N1–N12) that any primitive solution must satisfy, organized as: counting congruences → sharp $p$-adic laws → the aligned-case master congruence → residue filters → metric bounds → factorization identities → parametric obstructions. All congruence claims below were machine-verified (residue sets, root groups, valuation identities, CRT counts, local witnesses); the verification log is in the Appendix. The verdict (Section 9) is honest: **no known condition, and no finite set of congruences, forces non-existence** — we prove the congruence system is locally solvable at every modulus, so any non-existence proof must be global.

**Notation.** Indices $i_2,i_3,i_7\in\{1,\dots,5\}$ denote the (unique, by N1) positions with $a_{i_2}$ odd, $3\nmid a_{i_3}$, $7\nmid a_{i_7}$, and we write $a_o=a_{i_2}$, $a_T=a_{i_3}$, $a_S=a_{i_7}$.

---

## 1. The counting congruences (mod 7, 8, 9)

Since $\varphi(7)=\varphi(9)=6$ and odd squares are $\equiv1\pmod8$ (all verified by enumeration):

$$x^6\bmod 7\in\{0,1\},\qquad x^6\bmod 9\in\{0,1\},\qquad x^6\bmod 8\in\{0,1\}.$$

**N1 (three distinguished indices).** *In any primitive solution:*

- $7\nmid B$, and **exactly one** $a_i$ is coprime to $7$ (the other four are divisible by $7$);
- $3\nmid B$, and **exactly one** $a_i$ is coprime to $3$ (four divisible by $3$);
- $B$ is odd, and **exactly one** $a_i$ is odd (four even).

*Proof.* If $p\in\{2,3,7\}$ divided $B$, the sum of five residues in $\{0,1\}$ would have to be $\equiv0$, forcing **all five** $a_i$ to be divisible by $p$ — contradicting primitivity. So $B^6\equiv1$, and a sum of five $\{0,1\}$-residues equals $1$ only as $1+0+0+0+0$. Machine check: exactly $6$ of $32$ residue 5-tuples pass each modulus (the all-zero tuple plus the five single-unit tuples). $\blacksquare$

**N2 (divisibility corollaries).** By inclusion–exclusion over $\{i_2,i_3,i_7\}$:

- $\gcd(B,42)=1$;
- **at least two of the $a_i$ are divisible by $42$**, hence $a_5\ge 84$ and $B\ge85$;
- at least three of the $a_i$ are divisible by each of $6$, $14$, $21$;
- the **aligned case** $i_2=i_3=i_7$ is extremal: one term coprime to $42$, four terms divisible by $42$, hence $a_5\ge168$.

---

## 2. The sharp $p$-adic laws (the descent engine)

N1 is only the mod-$p$ shadow. The four non-distinguished terms contribute $0\pmod{p^6}$, so the distinguished term must match $B^6$ modulo $p^6$. The sixth-power unit root groups are small and completely explicit (verified for all $k\le12$):

- $x^6\equiv1\pmod{2^k}\iff x\equiv\pm1\pmod{2^{k-1}}$ — four roots $\{\pm1,\ \pm1+2^{k-1}\}$ (e.g. $\{1,31,33,63\}$ mod $64$);
- $x^6\equiv1\pmod{3^k}\iff x\equiv\pm1\pmod{3^{k-1}}$ — six roots (only $\pm1$ survive past $3^3$: the extra roots mod $27$ such as $8$ fail to lift, $8^6\equiv28\pmod{81}$);
- $x^6\equiv1\pmod{7^k}$ has exactly six roots, the Teichmüller lifts of the six unit classes mod $7$; mod $7^6=117649$ they are

$$\zeta_c\in\{1,\ 34967,\ 34968,\ 82681,\ 82682,\ 117648\},\qquad \zeta_c\equiv c\pmod 7,\quad \zeta_{-c}\equiv-\zeta_c\pmod{7^6}.$$

**N3 (fundamental congruences).**

$$B\equiv\pm a_o\pmod{32},\qquad B\equiv\pm a_T\pmod{243},\qquad B\equiv\zeta_c\,a_S\pmod{117649},\quad c\equiv B\,a_S^{-1}\pmod 7.$$

**N4 (self-improving descent).** Let $m=\min v_2$, $t=\min v_3$, $u=\min v_7$ taken over the respective four non-distinguished terms. Then

$$B\equiv\pm a_o\pmod{2^{6m-1}},\qquad B\equiv\pm a_T\pmod{3^{6t-1}},\qquad B\equiv\zeta_c\,a_S\pmod{7^{6u}}.$$

*Mechanism* (verified numerically on thousands of random pairs): the exact valuation identities, valid whenever $B,a$ are units at the relevant prime,

$$v_2(B^6-a^6)=v_2(B-a)+v_2(B+a),\qquad v_3(B^6-a^6)=v_3(B\mp a)+1,\qquad v_7(B^6-a^6)=v_7(B-\zeta_c a),$$

combined with $v_p(B^6-a^6)\ge 6\cdot(\text{min valuation})$, since $B^6-a^6$ is a sum of four sixth powers each divisible by $p^{6(\cdot)}$. $\blacksquare$

**Descent content.** Since $a_S<B\le 5^{1/6}a_S\approx1.3077\,a_S$ (N8), $B$ is squeezed into a short real interval while being pinned to a $7$-adic ray through $a_S$ of precision $7^{6u}$. On the $\zeta=1$ branch, $B\equiv a_S\pmod{7^{6u}}$ with $B>a_S$ forces $B\ge a_S+7^{6u}$, i.e. $B\gtrsim4.2\cdot7^{6u}$; on the $\zeta\equiv-1$ branch, $B+a_S\ge7^{6u}$. Extra divisibility in the small terms therefore explodes $B$: already $u=2$ (i.e. $49$ dividing the four small terms) pushes $B$ to order $7^{12}\approx1.4\times10^{10}$ (branch-dependent constants, same order of magnitude). Solutions, if they exist, live on thin $p$-adic arcs — which is exactly why they are invisible to naive search.

**Empirical confirmation one level up.** Known $(6,1,7)$ solutions obey the analogous law and sit on the $\zeta\equiv-1$ branch: de Smedt's

$$63631^6=54138^6+54018^6+48090^6+39088^6+25263^6+23268^6+1344^6$$

has $B+a=63631+54018=117649=7^6$ **exactly** (verified, identity included; six of the seven terms are divisible by $7$, $B\equiv1\pmod7$).[^3^]

---

## 3. The aligned case: the $42^6$ master congruence

**N5.** *If $i_2=i_3=i_7=*$ (one term coprime to $42$, four divisible by $42$), then*

$$42^6\mid B^6-a_*^6,\qquad\text{i.e.}\qquad B\equiv\omega\,a_*\pmod{42^6},\qquad 42^6=5\,489\,031\,744,$$

*where $\omega$ ranges over exactly $4\times6\times6=144$ residue classes* (CRT product of the root groups in §2; count verified). Consequences:

- on the $\omega=1$ branch, $B\ge a_*+42^6>5.4\times10^9$;
- given $a_*$ and a branch, $B$ is pinned to **one residue class mod $42^6$**;
- $B^6-a_*^6$ must split as four *distinct* multiples of $42$ raised to the sixth power, with minimal possible sum $42^6(1^6+2^6+3^6+4^6)=4890\cdot42^6$.

This is the strongest single structural law known for the problem — the exact analogue of the mod-$(k+1)^k$ bottleneck observed for $(6,1,6)$.

---

## 4. Residue filters at primes $p\equiv1\pmod6$ (the search sieve)

For $p\equiv1\pmod6$ the sixth-power residues $R_p=\{0\}\cup(\mathbb F_p^\times)^6$ have only $1+(p-1)/6$ elements (verified):

$$R_{13}=\{0,1,12\},\quad R_{19}=\{0,1,7,11\},\quad R_{37}=\{0,1,10,11,26,27,36\}.$$

**N6 (signature conditions).** $\sum_i r_i\in R_p$ with each $r_i\in R_p$. Sharply:

- **mod 13:** writing $p_+=\#\{i:a_i^6\equiv1\}$, $p_-=\#\{i:a_i^6\equiv-1\}$ (among terms with $13\nmid a_i$), one needs $p_+-p_-\in\{-1,0,1\}$; and $13\mid B\iff p_+=p_-$ (contrast with $p=7$: $13\mid B$ is *allowed*). Admissible $(p_+,p_-)$: $(0,1),(1,0),(1,1),(2,1),(1,2),(2,2),(3,2),(2,3)$ (verified by enumeration).
- **mod 19, 37, 31, 43, …:** the sum of five residues must again be a residue; measured pass rates $141/243$, $271/1024$, $5881/16807$ (verified). Same mechanism that powers the local test in Newton–Rouse (§8).
- **mod 5 / 25:** $x^6\equiv x^2\pmod5$ gives the $\{0,\pm1\}$ signature filter one level down (pass rate mod 25: $81401/161051$).

**N7 (combined density).** By CRT the independent-prime conditions thin the naive search space by

$$\frac{5}{64}\cdot\frac{20}{729}\cdot\frac{180}{117649}\cdot\frac{141}{243}\cdot\frac{271}{1024}\ \approx\ 5.0\times10^{-7}$$

(primes $2,3,7,13,19$ only) — the Ward (1948) → Lander–Parkin–Selfridge → Resta–Meyrignac "congruential restraints" machinery that pushed the exhaustive bound to $730{,}000$.[^1^][^3^]

---

## 5. Metric conditions and the counting heuristic

**N8 (size window).**

$$B\cdot5^{-1/6}\le a_5<B,\qquad\text{i.e.}\qquad 0.7647\,B\le a_5\le B-1,$$

and recursively $a_j\ge\big((B^6-\sum_{i>j}a_i^6)/j\big)^{1/6}$ — the window used in every serious search.

**N9 (borderline heuristic).** The number of representations of $N$ as a sum of five sixth powers grows like $N^{-1/6}$; requiring $N=B^6$ gives expected solution count $\sum_{B\le X}c\,B^{-1}\sim c\log X$ — *exactly* at the convergence/divergence threshold, mirroring the LPS boundary $k=m+n$. With the sieve factor $\sim10^{-7}$ of N7 folded into $c$, the first solution, if one exists, can easily lie beyond any computationally reachable $X$. Both existence and non-existence are consistent with all known heuristics.

---

## 6. Factorization identities

**N10.** For each $i$:

$$\underbrace{(B-a_i)(B+a_i)(B^2-Ba_i+a_i^2)(B^2+Ba_i+a_i^2)}_{=\,B^6-a_i^6}\;=\;\sum_{j\ne i}a_j^6,$$

so each left-hand factor divides a sum of four sixth powers, and:

- the quadratic factors are norm forms of $\mathbb Z[\omega]$: any prime $p\equiv2\pmod3$ dividing $B^2\pm Ba_i+a_i^2$ divides $\gcd(B,a_i)$; hence when $\gcd(B,a_i)=1$, every $p\equiv2\pmod3$ occurs to even order in $B^2\pm Ba_i+a_i^2$;
- the valuations of the left side are forced by N3–N4 (exactly one of $B\pm a_o$ is $\equiv2\pmod4$; the other carries the full $v_2\ge 6m-1$);
- two global reformulations: $(a_1^3)^2+\cdots+(a_5^3)^2=(B^3)^2$ and $(a_1^2)^3+\cdots+(a_5^2)^3=(B^2)^3$ — a solution is simultaneously a representation of a cube as five squares of cubes, and of a square as five cubes of squares.

---

## 7. Parametric obstructions

**N11 (Newton–Girard collapse — verified symbolically).** If a solution additionally satisfied any of the classical multigrade/PTE power-sum equalities $\sum_i a_i^j=B^j$ for $j=1,\dots,5$, then Newton's identities give the elementary symmetric functions $(e_1,e_2,e_3,e_4,e_5)=(B,0,0,0,0)$, so the $a_i$ are the roots of $x^5-Bx^4=x^4(x-B)$, i.e. $\{0,0,0,0,B\}$ — impossible for positive integers. **Hence no Choudhry/Subba-Rao/Moessner-type multigrade or ideal Tarry–Escott identity can generate (6,1,5).** This closes the parametric route that solved $(6,3,3)$ (Subba Rao 1934, parametric; also Brudno, Delorme) and $(5,1,5)$ (Sastry); any parametric attack must use genuinely asymmetric identities, and none is known — every known sixth-power parametric family lives in a balanced signature.[^3^][^5^]

**N12 (geometric obstruction to general theory).** The solution variety $X:\ x_1^6+\cdots+x_5^6=x_6^6\subset\mathbb P^5$ is a smooth sextic fourfold with $K_X=(6-6)H=0$ — a **Calabi–Yau fourfold**, sitting exactly at the Fermat threshold. So no Faltings-type finiteness applies; rational points are heuristically log-sparse (N9); and neither "infinitely many" nor "none" is provable by current geometric methods. (Contrast: the two-term variety of Newton–Rouse below is a genus-10 curve, and the $(6,2,2)$ surface is of general type — both *harder* geometries where non-existence machinery has teeth; see §8.)

---

## 8. The Newton–Rouse machinery and its bearing on (6,1,5)

Reference: A. Newton and J. Rouse, *Integers that are sums of two rational sixth powers*, INTEGERS, arXiv:2101.09390.[^4^] This is the "nearby problem" whose methods are most relevant here.

**What they prove.** $164634913=(44/5)^6+(117/5)^6$ is the smallest positive integer that is a sum of two *rational* sixth powers but not a sum of two *integer* sixth powers; and infinitely many integers have this property (their Theorem 2 uses the mod-13 obstruction: an integer $\equiv5\pmod{13}$ is never a sum of two integer sixth powers, since two-term sums of residues in $R_{13}=\{0,\pm1\}$ lie in $\{0,\pm1,\pm2\}$).

**Their method, step by step, versus our conditions.**

1. **Local solvability criterion (their Thm. 5).** For $C_k:x^6+y^6=kz^6$ ($k$ sixth-power free), $C_k$ is locally solvable iff it has points over $\mathbb Q_p$ for all $p<400$ and every odd prime $p\mid k$ satisfies $p\equiv1\pmod4$. The proof uses exactly our residue calculus: Hasse's bound $\#C_k(\mathbb F_p)\ge p+1-20\sqrt p>0$ for $p>400$ (genus 10), Hensel lifting, and $x^6+y^6\equiv x^2+y^2\pmod p$ for $p\mid k$. Their sieve congruences $k\equiv1,2\pmod7$, $k\equiv1,2\pmod8$, $k\equiv1,2\pmod9$ are the two-term shadows of our N1 (same sets $R_7=R_8=R_9=\{0,1\}$), and their cached sums of two sixth powers mod $p$, $13\le p\le400$, $p\equiv1\pmod6$, are precisely our N6 filters.
2. **The construction.** The representation $(44/5)^6+(117/5)^6$ was found by seeking $x^6+y^6\equiv0\pmod{5^6}$ with $xy^{-1}$ of order $4$ in $(\mathbb Z/5^6\mathbb Z)^\times$ ($q=1068$), then LLL-reducing the lattice $\{(x,y):y\equiv qx\pmod{5^6}\}$ — the same "work modulo $p^6$" principle as our N3–N5.
3. **The non-existence engine.** $C_k$ has genus $10$, and $\mathrm{Jac}(C_k)$ decomposes up to isogeny into a product of **ten elliptic curves, all with $j=0$** ($E_a:y^2=x^3+a$), with explicit quotient maps to $E_k,E_{4k},E_{-k^2},E_{16k^2},E_{k^3},E_{-4k^4}$. For each of the $111625$ locally solvable $k<164634913$, they either find a rank-0 elliptic quotient or apply the **Mordell–Weil sieve** (Bruin–Stoll) with a finite-index subgroup of the Mordell–Weil group to show $C_k(\mathbb Q)=\varnothing$. This is currently the *only* method that has proved non-existence of rational points on a Fermat-type sixth-power variety.
4. **Adjacent data point.** They note the surface $X:a^6+b^6=c^6+d^6$ (i.e. $(6,2,2)$) is of **general type**, Bombieri–Lang predicts finitely many rational points off genus $\le1$ curves, and Ekl's search found no nontrivial $(6,2,2)$ solutions with $a^6+b^6<7.25\times10^{24}$.

**Transfer to (6,1,5).**

- The *local* half of their toolkit is literally identical to ours (same residue sets, same $p\equiv1\bmod6$ sieve, same mod-$p^6$ lifting). N1–N7 are the five-term continuation of their Section 4.
- The *global* half (Jacobian decomposition + MW sieve) does not transfer directly: our variety is a Calabi–Yau fourfold, not a genus-10 curve. But it suggests the only credible non-existence framework: **fiber the fourfold into curves** (fix ratios of four of the five variables, or intersect with rational hyperplanes/curves), decompose Jacobians, and sieve curve-by-curve. No fibration is known that makes this work for $(6,1,5)$ — identifying one is, in our view, the most promising route toward a non-existence proof.
- The hardness contrast is sharp: $(6,2,1)$ lives on a genus-10 curve (general type — finite by Faltings), $(6,2,2)$ on a general-type surface (Bombieri–Lang-finite heuristically), while $(6,1,5)$ sits at $K_X=0$ — the first case where *no* finiteness theorem applies, and exactly the LPS threshold.

---

## 9. Which of these conditions force non-existence?

**None — and congruences provably never can.**

1. **No local obstruction exists.** Primitive witnesses of the full congruence system exist at every relevant modulus (verified):

   | modulus | witness $(a_1,\dots,a_5;B)$ | check |
   |---|---|---|
   | $42^6$ | $(1,42,84,126,168;\ 1)$ | $1+0+0+0+0\equiv1$, gcd $=1$ |
   | $13$ | $(1,1,1,2,2;\ 1)$ | $1+1+1+64+64=131\equiv1=1^6$ |
   | $19$ | $(1,1,1,1,2;\ 4)$ | $4+64=68\equiv11\equiv4^6$ |
   | $37$ | $(1,1,1,6,6;\ 1)$ | $3+2\cdot46656\equiv3-2\equiv1=1^6$ |

   Since the system is locally solvable everywhere, **no finite set of congruences can prove non-existence.** Any non-existence proof must be global: a descent producing a smaller solution (none is known), or a height/Jacobian argument on the fourfold (currently out of reach; see §8 on the Newton–Rouse sieve as the model).

2. **The only proved non-existence statements are:**
   (a) $B\le730{,}000$ excluded exhaustively (2002);[^1^]
   (b) the strengthened simultaneous system $\sum_i a_i^j=B^j$, $j\le6$, is impossible (N11);
   (c) the two-term case $(6,1,2)$ — Fermat's Last Theorem for exponent $3$;
   (d) two rational sixth powers suffice for integers like $164634913$ where two integer sixth powers provably do not (Newton–Rouse).[^4^]

3. **What the conditions do achieve:** they collapse the search to $\sim10^{-7}$ of the naive space (N7); pin $B$ to one of $144$ classes mod $42^6$ in the aligned case (N5); force any solution with extra divisibility into ranges $\gtrsim10^{10}$ (N4); and explain both why nothing has been found and how to look: iterate the distinguished term $a_*$ and the branch $\zeta$, read off $B$ candidates from N3/N5 inside the window N8, and test whether $B^6-a_*^6$ splits as a sum of four sixth powers through the N6 sieve.

---

## Appendix: machine-verification log

All congruence claims in this note were checked by direct enumeration (Python, arbitrary precision):

- **Residue sets** $R(m)=\{x^6\bmod m\}$ for $m=7,8,9,13,16,19,25,27,37,49,64,81,128$: as quoted in §1, §4 (e.g. $R(64)=\{0,1,9,17,25,33,41,49,57\}$).
- **Counting filters:** $6/32$ passing tuples at $m=7,8,9$ (the all-zero tuple plus five single-unit tuples — N1); $141/243$ at $13$; $271/1024$ at $19$; $5881/16807$ at $37$; $81401/161051$ at $25$.
- **Root groups:** $x^6\equiv1\pmod{2^k}$ has roots $\{\pm1,\pm1+2^{k-1}\}$ for $k=3,\dots,12$; $x^6\equiv1\pmod{3^k}$ iff $x\equiv\pm1\pmod{3^{k-1}}$ for $k=2,\dots,10$; six Teichmüller roots mod $7^k$ for $k=1,\dots,6$ reducing bijectively to the unit classes mod $7$, with $\zeta_c=c^{7^5}\bmod 7^6$ giving $\{1,34967,34968,82681,82682,117648\}$.
- **Valuation identities** of N4: verified on $2000$ random pairs each for $p=2,3,7$ (no failures).
- **CRT count:** $4\times6\times6=144$ root classes mod $42^6=5489031744$ (N5).
- **Local witnesses** of §9, table: all verified exactly.
- **de Smedt $(6,1,7)$ example:** identity verified; $63631+54018=117649=7^6$; six terms divisible by $7$.
- **Newton–Girard:** symbolic solution of the power-sum recurrence gives $(e_1,\dots,e_5)=(B,0,0,0,0)$ and quintic $x^4(x-B)$ (N11).
- Constants: $5^{1/6}=1.307660486\ldots$, $1/5^{1/6}=0.76472\ldots$; combined sieve density $5.0\times10^{-7}$ (N7).

A full independent test suite (re-derivation of every congruence from first principles, plus a branch-by-branch search scaffold) is planned as a separate step.

---

[^1^]: *Euler's sum of powers conjecture*, Wikipedia — including the statement that there are no solutions for $k=6$ with final term $\le730{,}000$ (known since 2002). https://en.wikipedia.org/wiki/Euler%27s_sum_of_powers_conjecture
[^2^]: *Lander, Parkin, and Selfridge conjecture* ($m+n\ge k$), Wikipedia. https://en.wikipedia.org/wiki/Lander,_Parkin,_and_Selfridge_conjecture
[^3^]: J.-C. Meyrignac (ed.), *Records of equal sums of like powers* — includes the 1966 Lander–Parkin–Selfridge $(6,1,7)$ solution $1141^6=\cdots$, de Smedt's $63631^6=\cdots$, and the absence of any known $(6,1,n)$, $n<7$. http://euler.free.fr/records.htm
[^4^]: A. Newton and J. Rouse, *Integers that are sums of two rational sixth powers*, INTEGERS, arXiv:2101.09390v2 [math.NT]. https://arxiv.org/abs/2101.09390
[^5^]: G. Resta and J.-C. Meyrignac, *The smallest solutions to the Diophantine equation $x^6+y^6=a^6+b^6+c^6+d^6+e^6$*, Math. Comp. 72 (2003), 1051–1054.
[^6^]: L. J. Lander, T. R. Parkin and J. L. Selfridge, *A survey of equal sums of like powers*, Math. Comp. 21 (1967), 446–459.
