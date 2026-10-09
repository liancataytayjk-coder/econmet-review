# Exercise 1 — Worked Answers

These answers use only what is in Lesson 1 (Linear Regression Analysis), Lecture 1A (Concept of Regression Analysis and the Error Term) and Lecture 1B (Regression in Matrix Notation).

Assumptions taken from Lesson 1: the X's are fixed constants and the εᵢ are independent N(0, σ²). So E(ε) = 0 and the variance-covariance matrix of ε is σ²I.

---

## 1. The model in expanded matrix form

Model: Tradeᵢ = β₀ + β₁Exportᵢ + β₂Importᵢ + β₃ERᵢ + β₄Tariffᵢ + β₅Costᵢ + εᵢ, for i = 1, 2, …, n

The model has 6 parameters (p = 6), so X has 6 columns: a column of 1's for the constant, then one column for each regressor.

```
Y = Xβ + ε

[ Trade₁ ]   [ 1  Export₁  Import₁  ER₁  Tariff₁  Cost₁ ] [ β₀ ]   [ ε₁ ]
[ Trade₂ ]   [ 1  Export₂  Import₂  ER₂  Tariff₂  Cost₂ ] [ β₁ ]   [ ε₂ ]
[ Trade₃ ] = [ 1  Export₃  Import₃  ER₃  Tariff₃  Cost₃ ] [ β₂ ] + [ ε₃ ]
[   ⋮    ]   [ ⋮     ⋮        ⋮      ⋮      ⋮       ⋮   ] [ β₃ ]   [ ⋮  ]
[ Tradeₙ ]   [ 1  Exportₙ  Importₙ  ERₙ  Tariffₙ  Costₙ ] [ β₄ ]   [ εₙ ]
                                                          [ β₅ ]
  (n×1)                        (n×6)                       (6×1)    (n×1)
```

- **Y** (n×1): the n observations of Trade
- **X** (n×6): constants. Column 1 is all 1's (for β₀), and columns 2–6 hold Export, Import, ER, Tariff and Cost.
- **β** (6×1): the regression parameters β₀ … β₅
- **ε** (n×1): the random error terms

Lecture 1B writes the same model as **Y = X′β + ε** and prints the data matrix transposed. That is the 6×n matrix below, with a row of 1's followed by one row per variable:

```
     [ 1        1        …  1       ]
     [ Export₁  Export₂  …  Exportₙ ]
X′ = [ Import₁  Import₂  …  Importₙ ]      (6×n)
     [ ER₁      ER₂      …  ERₙ     ]
     [ Tariff₁  Tariff₂  …  Tariffₙ ]
     [ Cost₁    Cost₂    …  Costₙ   ]
```

Both layouts contain the same elements. For the multiplication to work, the n×6 matrix is the one that multiplies β.

---

## 2. The solution matrix

The least-squares solution is **β̂ = (X′X)⁻¹X′Y**.

Where it comes from (Lesson 1):
ε′ε = (Y − Xβ)′(Y − Xβ) = Y′Y − 2β′X′Y + β′X′Xβ

∂(ε′ε)/∂β = −2X′Y + 2X′Xβ = 0  →  X′Xβ̂ = X′Y  →  **β̂ = (X′X)⁻¹X′Y**

*(The handout prints the derivative as "X′Y + X′Xβ". Writing it as −2X′Y + 2X′Xβ is the same condition once you set it equal to zero.)*

### 2a. How the elements of X′X are derived

X′ is (6×n) and X is (n×6), so X′X is (6×6). The element in row j, column k is
(row j of X′) · (column k of X) = Σᵢ (variable j)ᵢ × (variable k)ᵢ

```
      [ 1    1   … 1  ] [ 1 Exp₁ Imp₁ ER₁ Tar₁ Cost₁ ]
      [ Exp₁ Exp₂ … Expₙ] [ 1 Exp₂ Imp₂ ER₂ Tar₂ Cost₂ ]
X′X = [ Imp₁ Imp₂ … Impₙ] [ ⋮   ⋮    ⋮    ⋮   ⋮    ⋮  ]
      [ ER₁  ER₂  … ERₙ ] [ 1 Expₙ Impₙ ERₙ Tarₙ Costₙ ]
      [ Tar₁ Tar₂ … Tarₙ]
      [ Cost₁ Cost₂ … Costₙ]
```

```
      [ n        ΣExp        ΣImp        ΣER        ΣTar        ΣCost      ]
      [ ΣExp     ΣExp²       ΣExp·Imp    ΣExp·ER    ΣExp·Tar    ΣExp·Cost  ]
X′X = [ ΣImp     ΣImp·Exp    ΣImp²       ΣImp·ER    ΣImp·Tar    ΣImp·Cost  ]
      [ ΣER      ΣER·Exp     ΣER·Imp     ΣER²       ΣER·Tar     ΣER·Cost   ]
      [ ΣTar     ΣTar·Exp    ΣTar·Imp    ΣTar·ER    ΣTar²       ΣTar·Cost  ]
      [ ΣCost    ΣCost·Exp   ΣCost·Imp   ΣCost·ER   ΣCost·Tar   ΣCost²     ]   (6×6)
```

- Row 1 × column 1: 1·1 + 1·1 + … + 1·1 = **n**
- Row 1 × column j: 1·X₁ + … + 1·Xₙ = **ΣX** (the sum of that variable)
- Row j × column j: **ΣX²** (the sum of squares, on the diagonal)
- Row j × column k: **ΣXⱼXₖ** (the sum of cross-products, off the diagonal)
- X′X is **symmetric**: the element in row j, column k equals the one in row k, column j.

### 2b. How the elements of X′Y are derived

X′ is (6×n) and Y is (n×1), so X′Y is (6×1). Element j = Σᵢ (variable j)ᵢ × Tradeᵢ

```
      [ 1    1   … 1   ] [ Trade₁ ]   [ ΣTrade        ]
      [ Exp₁ Exp₂ … Expₙ ] [ Trade₂ ]   [ ΣExp·Trade    ]
X′Y = [ Imp₁ Imp₂ … Impₙ ] [   ⋮    ] = [ ΣImp·Trade    ]
      [ ER₁  ER₂  … ERₙ  ] [ Tradeₙ ]   [ ΣER·Trade     ]
      [ Tar₁ Tar₂ … Tarₙ ]              [ ΣTar·Trade    ]
      [ Cost₁ Cost₂ … Costₙ]            [ ΣCost·Trade   ]   (6×1)
```

### 2c. Final form of the solution matrix

```
[ β̂₀ ]   [ n      ΣExp       ΣImp       ΣER       ΣTar       ΣCost     ]⁻¹ [ ΣTrade      ]
[ β̂₁ ]   [ ΣExp   ΣExp²      ΣExp·Imp   ΣExp·ER   ΣExp·Tar   ΣExp·Cost ]   [ ΣExp·Trade  ]
[ β̂₂ ] = [ ΣImp   ΣImp·Exp   ΣImp²      ΣImp·ER   ΣImp·Tar   ΣImp·Cost ]   [ ΣImp·Trade  ]
[ β̂₃ ]   [ ΣER    ΣER·Exp    ΣER·Imp    ΣER²      ΣER·Tar    ΣER·Cost  ]   [ ΣER·Trade   ]
[ β̂₄ ]   [ ΣTar   ΣTar·Exp   ΣTar·Imp   ΣTar·ER   ΣTar²      ΣTar·Cost ]   [ ΣTar·Trade  ]
[ β̂₅ ]   [ ΣCost  ΣCost·Exp  ΣCost·Imp  ΣCost·ER  ΣCost·Tar  ΣCost²    ]   [ ΣCost·Trade ]
 (6×1)                         (X′X)⁻¹  (6×6)                                X′Y  (6×1)
```

This follows the same pattern as the 2-variable and 3-variable cases in Lecture 1B, extended to 6 parameters.

---

## 3. Regression through the origin: Yᵢ = β₁Xᵢ + εᵢ

### Estimator of β₁ (OLS)

The residual is eᵢ = Yᵢ − b₁Xᵢ, so

RSS = Σeᵢ² = Σ(Yᵢ − b₁Xᵢ)²

The first-order condition:

∂RSS/∂b₁ = −2ΣXᵢ(Yᵢ − b₁Xᵢ) = 0
→ ΣXᵢYᵢ − b₁ΣXᵢ² = 0   *(this model has only one normal equation)*

> **β̂₁ = b₁ = ΣXᵢYᵢ / ΣXᵢ²**

The second derivative is ∂²RSS/∂b₁² = 2ΣXᵢ² > 0, so this is a minimum.

Unlike the model with a constant, there is no b₀ = Ȳ − b₁X̄, and the formula uses raw sums rather than deviations from the mean.

### Variance of β̂₁

Substitute Yᵢ = β₁Xᵢ + εᵢ:

β̂₁ = ΣXᵢ(β₁Xᵢ + εᵢ)/ΣXᵢ² = β₁ + ΣXᵢεᵢ / ΣXᵢ²

**Unbiased:** E(β̂₁) = β₁ + ΣXᵢE(εᵢ)/ΣXᵢ² = β₁, because the X's are fixed and E(εᵢ) = 0.

**Variance:**

Var(β̂₁) = E[(β̂₁ − β₁)²] = E[(ΣXᵢεᵢ)²] / (ΣXᵢ²)²

The εᵢ are independent, so E(εᵢεⱼ) = 0 for i ≠ j and E(εᵢ²) = σ². That leaves only the squared terms:

E[(ΣXᵢεᵢ)²] = ΣXᵢ²E(εᵢ²) = σ²ΣXᵢ²

> **Var(β̂₁) = σ²ΣXᵢ² / (ΣXᵢ²)² = σ² / ΣXᵢ²**

**Variance estimator:** σ² is unknown, so replace it with its unbiased estimate from the residuals. Only 1 parameter is estimated (p = 1; see the unbiasedness result in Item 4), so

s² = Σeᵢ² / (n − 1)   and   **Var̂(β̂₁) = s² / ΣXᵢ² = [Σ(Yᵢ − β̂₁Xᵢ)² / (n − 1)] / ΣXᵢ²**

---

## 4. Maximum likelihood estimation of β and σ² in Y = Xβ + ε

Each εᵢ ~ N(0, σ²), independent. So the likelihood is the product of normal densities:

L = Πᵢ (1/(√(2π)σ)) e^(−εᵢ²/2σ²) = (2πσ²)^(−n/2) · exp[ −(1/2σ²) Σεᵢ² ]

Here Σεᵢ² = ε′ε = (Y − Xβ)′(Y − Xβ).

**Log-likelihood** (easier to maximize, and it has the same maximum):

ln L = −(n/2)ln(2π) − (n/2)ln σ² − (1/2σ²)(Y − Xβ)′(Y − Xβ)

Expanding the last term: (Y − Xβ)′(Y − Xβ) = Y′Y − 2β′X′Y + β′X′Xβ

### MLE of β

∂lnL/∂β = −(1/2σ²)(−2X′Y + 2X′Xβ) = 0
→ X′Xβ̃ = X′Y

> **β̃ = (X′X)⁻¹X′Y**   *(the same result as OLS)*

Only the last term of ln L contains β. So maximizing L over β is the same as minimizing (Y − Xβ)′(Y − Xβ), which is the least-squares criterion.

### MLE of σ²

∂lnL/∂σ² = −n/(2σ²) + (1/(2σ⁴))(Y − Xβ̃)′(Y − Xβ̃) = 0

> **σ̃² = (Y − Xβ̃)′(Y − Xβ̃) / n = e′e / n = Σeᵢ² / n**

### Is β̃ unbiased? Yes

β̃ = (X′X)⁻¹X′Y = (X′X)⁻¹X′(Xβ + ε) = β + (X′X)⁻¹X′ε

E(β̃) = β + (X′X)⁻¹X′E(ε) = β, because X is constant and E(ε) = 0.

> **E(β̃) = β → unbiased.** (Its variance is Var(β̃) = (X′X)⁻¹X′(σ²I)X(X′X)⁻¹ = σ²(X′X)⁻¹.)

### Is σ̃² unbiased? No

1. Write the residuals as e = Y − Xβ̃ = [I − X(X′X)⁻¹X′]Y = MY. Here M = I − X(X′X)⁻¹X′.
2. M has these properties: MX = 0, M′ = M (symmetric) and MM = M (idempotent). So e = M(Xβ + ε) = Mε and e′e = ε′M′Mε = ε′Mε.
3. Take the expectation. e′e is a scalar, so it equals its own trace:
   E(e′e) = E[tr(ε′Mε)] = E[tr(Mεε′)] = tr[M·E(εε′)] = tr(M·σ²I) = σ²·tr(M)
4. Compute the trace of M:
   tr(M) = tr(Iₙ) − tr[X(X′X)⁻¹X′] = n − tr[(X′X)⁻¹X′X] = n − tr(I_p) = **n − p**
5. So E(e′e) = σ²(n − p), and

> **E(σ̃²) = E(e′e)/n = σ²(n − p)/n ≠ σ² → σ̃² is BIASED (it underestimates σ²).**

The unbiased estimator divides by the degrees of freedom instead of n:

> **s² = e′e / (n − p)**,   E(s²) = σ²

Here p is the number of parameters. In Item 1, p = 6, so s² = Σeᵢ²/(n − 6). In Item 3, p = 1, so s² = Σeᵢ²/(n − 1). The bias shrinks as n grows, because (n − p)/n → 1.

---

## 5. Computing b₀, b₁, b₂ with the 3-variable formula

Formula from Lecture 1B / Lesson 1:

```
[ b₀ ]   [ n     ΣX₁     ΣX₂   ]⁻¹ [ ΣY   ]
[ b₁ ] = [ ΣX₁   ΣX₁²    ΣX₁X₂ ]   [ ΣX₁Y ]
[ b₂ ]   [ ΣX₂   ΣX₁X₂   ΣX₂²  ]   [ ΣX₂Y ]
```

### Step 1 — Working table

|  Y | X₁ | X₂ | X₁² |  X₂²  | X₁X₂ | X₁Y | X₂Y  |
|---:|---:|---:|----:|------:|-----:|----:|-----:|
| 10 |  5 | 30 |  25 |   900 |  150 |  50 |  300 |
| 16 |  8 | 20 |  64 |   400 |  160 | 128 |  320 |
|  4 |  2 | 60 |   4 |  3600 |  120 |   8 |  240 |
|  8 |  4 | 80 |  16 |  6400 |  320 |  32 |  640 |
| 20 | 10 | 40 | 100 |  1600 |  400 | 200 |  800 |
| 14 |  7 | 50 |  49 |  2500 |  350 |  98 |  700 |
|  6 |  3 | 60 |   9 |  3600 |  180 |  18 |  360 |
| 12 |  6 | 30 |  36 |   900 |  180 |  72 |  360 |
| 18 |  9 | 70 |  81 |  4900 |  630 | 162 | 1260 |
|  2 |  1 | 60 |   1 |  3600 |   60 |   2 |  120 |
| **Σ = 110** | **55** | **500** | **385** | **28 400** | **2 550** | **770** | **5 100** |

n = 10

### Step 2 — Set up X′X and X′Y

```
        [  10     55     500  ]          [  110 ]
X′X =   [  55    385    2550  ]   X′Y =  [  770 ]
        [ 500   2550   28400  ]          [ 5100 ]
```

### Step 3 — Determinant of X′X (cofactor expansion along row 1)

|X′X| = 10(385·28400 − 2550²) − 55(55·28400 − 2550·500) + 500(55·2550 − 385·500)
     = 10(10 934 000 − 6 502 500) − 55(1 562 000 − 1 275 000) + 500(140 250 − 192 500)
     = 10(4 431 500) − 55(287 000) + 500(−52 250)
     = 44 315 000 − 15 785 000 − 26 125 000
     = **2 405 000**

### Step 4 — Cofactors / adjoint (X′X is symmetric, so the adjoint is symmetric too)

| Cofactor | Computation | Value |
|---|---|---:|
| C₁₁ | 385·28400 − 2550² | 4 431 500 |
| C₁₂ = C₂₁ | −(55·28400 − 2550·500) | −287 000 |
| C₁₃ = C₃₁ | 55·2550 − 385·500 | −52 250 |
| C₂₂ | 10·28400 − 500² | 34 000 |
| C₂₃ = C₃₂ | −(10·2550 − 55·500) | 2 000 |
| C₃₃ | 10·385 − 55² | 825 |

```
                     1        [ 4 431 500   −287 000   −52 250 ]
(X′X)⁻¹ =  ─────────────  ×   [  −287 000     34 000     2 000 ]
             2 405 000        [   −52 250      2 000       825 ]
```

### Step 5 — Multiply (X′X)⁻¹ by X′Y

b₀ = [4 431 500(110) − 287 000(770) − 52 250(5100)] / 2 405 000
   = [487 465 000 − 220 990 000 − 266 475 000] / 2 405 000 = 0 / 2 405 000 = **0**

b₁ = [−287 000(110) + 34 000(770) + 2 000(5100)] / 2 405 000
   = [−31 570 000 + 26 180 000 + 10 200 000] / 2 405 000 = 4 810 000 / 2 405 000 = **2**

b₂ = [−52 250(110) + 2 000(770) + 825(5100)] / 2 405 000
   = [−5 747 500 + 1 540 000 + 4 207 500] / 2 405 000 = 0 / 2 405 000 = **0**

> **b₀ = 0, b₁ = 2, b₂ = 0   →   Ŷ = 0 + 2X₁ + 0X₂ = 2X₁**

**Check with the normal equations:**
- ΣY = nb₀ + b₁ΣX₁ + b₂ΣX₂ → 110 = 10(0) + 2(55) + 0(500) = 110 ✓
- ΣX₁Y = b₀ΣX₁ + b₁ΣX₁² + b₂ΣX₁X₂ → 770 = 0 + 2(385) + 0 = 770 ✓
- ΣX₂Y = b₀ΣX₂ + b₁ΣX₁X₂ + b₂ΣX₂² → 5100 = 0 + 2(2550) + 0 = 5100 ✓

**Interpretation:** Y and X₁ are data set A from the Lesson 1 Activity, where every Y is exactly 2X₁. This is the "exact relationship" case from Lecture 1A: all points lie on a straight line. So the fit is perfect (every eᵢ = 0, so RSS = 0), the intercept is 0, and X₂ adds nothing (b₂ = 0).
