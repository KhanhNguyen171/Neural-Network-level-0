# Overview: Improving Transformer Models by Reordering their Sublayers

```text
┌───────────────────────────────────────────────────────────────┐
│                   TRANSFORMER DESIGN QUESTION                 │
└───────────────────────────────────────────────────────────────┘

     Transformer (Vaswani et al., 2017)

     A → F → A → F → A → F → A → F

     A = Self-Attention
     F = FeedForward

                     │
                     │
                     ▼

     Is alternating Attention and FFN really necessary?

                     │
                     ▼

┌───────────────────────────────────────────────────────────────┐
│                     PRESS ET AL. (2019)                       │
└───────────────────────────────────────────────────────────────┘

      Attention and FFN have different roles

      Attention
          ↓
      Communication

      FeedForward
          ↓
      Computation

                     │
                     ▼

      Therefore:

      Communicate First
      Compute Later

                     │
                     ▼

┌───────────────────────────────────────────────────────────────┐
│                    SANDWICH TRANSFORMER                       │
└───────────────────────────────────────────────────────────────┘


Traditional Transformer

Input
 │
 ▼
[A]
 │
 ▼
[F]
 │
 ▼
[A]
 │
 ▼
[F]
 │
 ▼
[A]
 │
 ▼
[F]
 │
 ▼
Output


Sandwich Transformer

Input
 │
 ▼
[A]
 │
 ▼
[A]
 │
 ▼
[A]
 │
 ▼
[A]
 │
 ▼
[A]
 │
 ▼

=========================
 Global Context Building
=========================

 │
 ▼

[F]
 │
 ▼
[F]
 │
 ▼
[F]
 │
 ▼
[F]
 │
 ▼
[F]
 │
 ▼

=========================
 Feature Processing
=========================

 │
 ▼
Output
```

---

## Communication vs Computation

```text
Traditional Transformer

Communicate
     ↓
Compute
     ↓
Communicate
     ↓
Compute
     ↓
Communicate
     ↓
Compute


Sandwich Transformer

Communicate
     ↓
Communicate
     ↓
Communicate
     ↓
Communicate

══════════════════

Compute
     ↓
Compute
     ↓
Compute
     ↓
Compute
```

---

## Information Flow Perspective

```text
Token 1
Token 2
Token 3
Token 4


Traditional Transformer

Layer 1
T1 ↔ T2
T2 ↔ T3

Layer 2
T3 ↔ T4

Layer 3
T1 ↔ T4

Communication occurs intermittently.


Sandwich Transformer

Attention 1
T1 ↔ T2 ↔ T3 ↔ T4

Attention 2
T1 ↔ T2 ↔ T3 ↔ T4

Attention 3
T1 ↔ T2 ↔ T3 ↔ T4

Attention 4
T1 ↔ T2 ↔ T3 ↔ T4

Global context forms before FFN processing.
```

---

## Graph-Theoretic Interpretation

```text
Transformer as Graph Neural Network


Attention
=
Message Passing


FeedForward
=
Node Update
```

Traditional Transformer:

```text
Message Passing
       ↓
Node Update
       ↓
Message Passing
       ↓
Node Update
```

Mathematically:

[
(M \circ U)^L
]

---

Sandwich Transformer:

```text
Message Passing
       ↓
Message Passing
       ↓
Message Passing
       ↓
Message Passing

══════════════════

Node Update
       ↓
Node Update
       ↓
Node Update
       ↓
Node Update
```

Mathematically:

[
U^m \circ M^n
]

---

## Sandwich Coefficient

The paper introduces:

[
c
]

called the **Sandwich Coefficient**.

Example:

```python
Encoder(
    dim = 512,
    depth = 12,
    sandwich_coef = 6
)
```

Conceptually:

```text
depth = 12


Standard

A F A F A F A F A F A F


Sandwich (c = 6)

A A A A A A
A/F A/F A/F
F F F
```

The paper reports that:

[
c \approx 6
]

works remarkably well across experiments.

---

## Computational Perspective

```text
Attention

Complexity

O(n²)



FeedForward

Complexity

O(n)
```

Observation:

```text
Most Transformer parameters
already reside in FFN.

Therefore:

Less Attention
More FeedForward

can preserve performance
while reducing cost.
```

---

## Evolution Toward Modern Transformers

```text
Transformer (2017)
         │
         ▼

Reordering Sublayers
(Press et al., 2019)

         │
         ▼

Sandwich Transformer

         │
         ▼

Flexible Layer Scheduling

         │
         ▼

DeepNet
PaLM
Parallel Transformer
Universal Transformer

         │
         ▼

x-transformers
```

---

## One-Sentence Summary

```text
Original Transformer

(A → F)^L


Sandwich Transformer

A^n → F^m


Core Idea:

Build global context first,
perform nonlinear computation later.
```
