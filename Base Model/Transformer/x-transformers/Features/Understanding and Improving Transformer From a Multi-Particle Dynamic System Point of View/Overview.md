# Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View

```text
┌───────────────────────────────────────────────────────────────┐
│                    TRANSFORMER AS A DYNAMIC SYSTEM            │
└───────────────────────────────────────────────────────────────┘

          Traditional View
                    │
                    ▼
          ┌───────────────────┐
          │   Attention + FFN │
          └───────────────────┘

                    │
                    │ Re-interpretation
                    ▼

┌───────────────────────────────────────────────────────────────┐
│                 MULTI-PARTICLE DYNAMIC SYSTEM                 │
└───────────────────────────────────────────────────────────────┘

          x₁        x₂        x₃        ...       xₙ
           ●────────●────────●────────────●
            ↖      ↗ ↖      ↗ ↖         ↗
             ↖    ↗   ↖    ↗   ↖       ↗
              ↖  ↗     ↖  ↗     ↖     ↗
               ↖↗       ↖↗       ↖   ↗

          Tokens behave like interacting particles

                       dX
                       ── = F(X)
                       dt

                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼

       F_att(X)                    F_ffn(X)

  Interaction Force           Internal Dynamics

 (Self-Attention)                (FeedForward)

                │                           │
                └─────────────┬─────────────┘
                              ▼

                    Total Dynamics

              dX
              ── = Fatt + Fffn
              dt
```

---

# 1. Physical Interpretation of Transformer

```text
                  PARTICLE SYSTEM

        x₁ ●──────────────►
             force

        x₂ ●──────────────►

        x₃ ●──────────────►


                Attention

      a₁₂ , a₁₃ , a₂₃ , ...

             determine

       how particles interact
```

Mathematical form:

$$
\frac{dx_i}{dt}= \sum_j a_{ij}g(x_j)
$$

where:

* particle = token
* force = attention weight
* system state = sequence representation

---

# 2. Role of Feed Forward Network

```text
      Transformer Community

         Attention
            ★★★★★

         FFN
            ★★


                ↓


      Dynamic System View

         Attention
            ★★★

         FFN
            ★★★

```

FFN is not merely a projection layer.

It represents:

$$
-\nabla U(x)
$$

which acts as the internal potential field of each particle.

```text
Particle

     ●
     │
     │ internal force
     ▼

    FFN
```

---

# 3. Original Transformer = Euler Solver

Standard Transformer layer:

```text
Input
  │
  ▼
Attention
  │
  ▼
FFN
  │
  ▼
Output
```

Equivalent to:

$$
x_{l+1}= x_l + F(x_l)
$$

which is exactly:

$$
x(t+\Delta t)= x(t)+\Delta tF(x(t))
$$

Euler Method.

---

```text
Numerical ODE Solver

        True Solution
        ╭─────────────╮
      ╭─╯             ╰─╮
    ╭─╯                 ╰─╮

Euler Approximation

    ╱
   ╱
  ╱
 ╱
```

Problem:

* First-order approximation
* Accumulated numerical error
* Deep transformers become less stable

---

# 4. Core Insight of the Paper

Instead of:

```text
F = Attention + FFN
```

view Transformer as:

```text
Operator A = FFN

Operator B = Attention
```

Need a better ODE solver.

---

# 5. Strang Splitting

Classical Numerical Analysis

```text
Euler

(A + B)

   │
   ▼

One Big Step
```

vs

```text
Strang Splitting

A/2
 │
 ▼
 B
 │
 ▼
A/2
```

Mathematically:

$$
e^{(A+B)\Delta t}
$$

becomes

$$
e^{A\Delta t/2} e^{B\Delta t} e^{A\Delta t/2}
$$

Higher accuracy.

---

# 6. Birth of Macaron Transformer

```text
STANDARD TRANSFORMER

 ┌─────────────┐
 │ Attention   │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │     FFN     │
 └─────────────┘
```

↓

```text
MACARON TRANSFORMER

 ┌─────────────┐
 │  FFN / 2    │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ Attention   │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  FFN / 2    │
 └─────────────┘
```

This is the main contribution.

---

# 7. Macaron Layer Dynamics

```text
x
│
│
├──► x + 0.5 FFN(x)
│
▼

x'

│
├──► x' + Attention(x')
│
▼

x''

│
├──► x'' + 0.5 FFN(x'')
│
▼

Output
```

Mathematically:

$$
x' = x+\frac12FFN(x)
$$

$$
x''= x'+Attention(x')
$$

$$
y= x''+\frac12FFN(x'')
$$

---

# 8. Why Half FFN?

Without scaling:

```text
FFN
 +
Attention
 +
FFN
```

FFN contribution doubles.

Therefore:

```text
0.5 FFN
 +
Attention
 +
0.5 FFN
```

ensures:

[0.5 + 0.5 = 1]

same total energy.

---

# 9. Theoretical Improvement

```text
Transformer

Euler Method
      │
      ▼
1st Order Accuracy

      O(Δt)
```

↓

```text
Macaron

Strang Splitting
       │
       ▼
2nd Order Accuracy

      O(Δt²)
```

---

# 10. Signal Propagation

```text
Standard Transformer

Signal
──────────╲
           ╲
            ╲
             ╲


Macaron

Signal
────────────────────
```

Macaron:

* less dissipation
* better gradient flow
* better information preservation

---

# 11. Connection to Neural ODE

```text
 Neural ODE

 dX/dt = F(X)
      │
      ▼

 Transformer
      │
      ▼

 Euler Solver
      │
      ▼

 Macaron
      │
      ▼

 Better ODE Solver
```

---

# 12. Influence on Later Architectures

```text
Transformer (2017)
          │
          ▼

Dynamic System View
(ICML 2020)
          │
          ▼

Macaron Transformer
          │
          ▼

Conformer
          │
          ▼

Modern Speech Models
```

---

# 13. Relationship with x-transformers

```python
Encoder(
    dim = 512,
    depth = 6,
    heads = 8,

    macaron = True
)
```

Internally:

```text
FFN/2
  ↓
Attention
  ↓
FFN/2
```

instead of:

```text
Attention
  ↓
FFN
```

---

# Final Mental Model

```text
                TRANSFORMER
                       │
                       ▼

         Multi-Particle Dynamic System

                       │
                       ▼

         dX/dt = Fatt + Fffn

                       │
                       ▼

          Transformer ≈ Euler Solver

                       │
                       ▼

            Numerical Error Exists

                       │
                       ▼

             Use Strang Splitting

                       │
                       ▼

                FFN/2
                  ↓
              Attention
                  ↓
                FFN/2

                       │
                       ▼

             MACARON TRANSFORMER

                       │
                       ▼

        Better Stability + Better Accuracy

                       │
                       ▼

               Conformer & Beyond
```
