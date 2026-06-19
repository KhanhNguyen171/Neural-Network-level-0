## Explicit Sparse Transformer Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                 EXPLICIT SPARSE TRANSFORMER                             │
│      Concentrated Attention Through Explicit Selection                  │
└──────────────────────────────────────────────────────────────────────────┘


                         Dense Self-Attention

                             Query (Q)
                                  │
                                  ▼

                     Similarity Computation

                           QKᵀ / √d
                                  │
                                  ▼

                     Attention Logits Matrix

          ┌─────────────────────────────────────┐
          │ 5.8   4.9   4.1   2.3   1.4   0.7  │
          └─────────────────────────────────────┘
                                  │
                                  │
                                  ▼

══════════════════════════════════════════════════════════════════════════════

                     Explicit Top-k Selection

                               k = 3

          ┌─────────────────────────────────────┐
          │ 5.8   4.9   4.1   2.3   1.4   0.7  │
          └─────────────────────────────────────┘

                     Keep Largest Scores

          ┌─────────────────────────────────────┐
          │ 5.8   4.9   4.1    X     X     X    │
          └─────────────────────────────────────┘

                     X = Removed Connection

══════════════════════════════════════════════════════════════════════════════

                           Sparse Mask

          ┌─────────────────────────────────────┐
          │  1     1     1     0     0     0    │
          └─────────────────────────────────────┘

                                  │
                                  ▼

                         Sparse Attention

          ┌─────────────────────────────────────┐
          │ 0.42  0.34  0.24  0.00  0.00  0.00 │
          └─────────────────────────────────────┘

                                  │
                                  ▼

                           Attention·V

                                  │
                                  ▼

                              Output
```


## Dense Attention vs Explicit Sparse Attention

```text
Transformer Attention

                         Query

     ┌─────┬─────┬─────┬─────┬─────┬─────┐
     ▼     ▼     ▼     ▼     ▼     ▼

    K1    K2    K3    K4    K5    K6

     │     │     │     │     │     │
     └─────┴─────┴─────┴─────┴─────┘

            All Connections Active
```

```text
Explicit Sparse Transformer

                         Query

           ┌────────┬────────┬────────┐
           ▼        ▼        ▼

          K1       K2       K3

          X        X        X

          K4       K5       K6


Only Top-k Connections Survive

Attention Graph Sparsified
```

### Core Principle

Dense Transformer:

```math
A = Softmax(QK^T)
```

Explicit Sparse Transformer:

```math
A = Softmax(TopKMask(QK^T,k))
```


## Information Flow Perspective

```text
Dense Transformer

Tokens
  │
  ▼

┌─────────────────────┐
│ Self Attention      │
└─────────────────────┘
  │
  ▼

Every Token
Attends To
Every Token


Number of Edges

N²
```

```text
Explicit Sparse Transformer

Tokens
  │
  ▼

┌─────────────────────┐
│ Top-k Selection     │
└─────────────────────┘
  │
  ▼

┌─────────────────────┐
│ Sparse Attention    │
└─────────────────────┘
  │
  ▼

Only Important
Connections Remain


Number of Edges

N × k

where

k << N
```

Example:

```text
N = 4096

Dense Attention

4096² = 16,777,216 edges


Top-k = 8

4096 × 8 = 32,768 edges


Reduction ≈ 512×
```


```text
                    EXPLICIT SPARSE TRANSFORMER

        Dense Attention                    Sparse Attention

      Q ───────────────► K1              Q ─────► K1
      │                 K2              │
      │                 K3              ├──────► K2
      │                 K4              │
      │                 K5              └──────► K3
      │                 K6

      All Connections                  Top-k Connections
             │                                 │
             ▼                                 ▼

      Softmax(QKᵀ)               Softmax(TopKMask(QKᵀ,k))
             │                                 │
             ▼                                 ▼

      Dense Distribution             Concentrated Distribution

 [0.22 0.19 0.17 0.15 0.12 0.15]

                 ↓

 [0.42 0.34 0.24 0.00 0.00 0.00]

                 ↓

       Less Noise + Better Alignment
       Lower Memory + Lower Compute
       Foundation of Sparse Transformers
```