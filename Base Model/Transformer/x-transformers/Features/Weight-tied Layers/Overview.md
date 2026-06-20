# Weight-Tied Layers: Overview

```text
┌───────────────────────────────────────────────────────────────────────┐
│                     WEIGHT-TIED LAYERS OVERVIEW                       │
│                                                                       │
│          "One Transformer Layer Reused Across Depth"                  │
└───────────────────────────────────────────────────────────────────────┘


                     STANDARD TRANSFORMER

      Input
        │
        ▼
 ┌──────────────┐
 │ Layer 1 (θ₁) │
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │ Layer 2 (θ₂) │
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │ Layer 3 (θ₃) │
 └──────────────┘
        │
        ▼
      Output


θ₁ ≠ θ₂ ≠ θ₃

Parameters:

Ptotal = L × P



═══════════════════════════════════════════════════════════════════════════



                    WEIGHT-TIED TRANSFORMER

                           Shared θ
                               │
                               ▼

            Input
            │
            ▼

┌─────────────────────┐
│ Transformer Block θ │
└─────────────────────┘
          │
          ▼

┌─────────────────────┐
│ Transformer Block θ │
└─────────────────────┘
          │
          ▼

┌─────────────────────┐
│ Transformer Block θ │
└─────────────────────┘
          │
          ▼

        Output


θ₁ = θ₂ = θ₃ = ... = θ

Parameters:

Ptotal = P



═══════════════════════════════════════════════════════════════════════════



                 DYNAMICAL SYSTEM INTERPRETATION

                 h₀
                  │
                  ▼

           h₁ = F(h₀;θ)

                  │
                  ▼

           h₂ = F(h₁;θ)

                  │
                  ▼

           h₃ = F(h₂;θ)

                  │
                  ▼

           h₄ = F(h₃;θ)

                  │
                  ▼

                ...

                  │
                  ▼

           hL = Fᴸ(h₀)



Core Equation:

hₗ₊₁ = F(hₗ ; θ)


Instead of learning:

F₁,F₂,F₃,...,Fᴸ

the model learns:

F

and applies it repeatedly.



═══════════════════════════════════════════════════════════════════════════



                    RELATION TO RNN

       Recurrent Neural Network

       hₜ = F(hₜ₋₁ ; θ)

               Time
                 ↑
                 │
                 │
                 │

────────────────────────────────────────

      Weight-Tied Transformer

      hₗ = F(hₗ₋₁ ; θ)

              Depth
                ↑
                │
                │
                │


RNN:
Weight sharing across time

Weight-Tied Transformer:
Weight sharing across depth



═══════════════════════════════════════════════════════════════════════════



                  INTERNAL SHARED BLOCK

        ┌──────────────────────────────┐
        │         LayerNorm            │
        ├──────────────────────────────┤
        │     Multi-Head Attention     │
        ├──────────────────────────────┤
        │         Residual             │
        ├──────────────────────────────┤
        │      Feed Forward Network    │
        ├──────────────────────────────┤
        │         Residual             │
        └──────────────────────────────┘

                   Shared Across

        Layer 1
        Layer 2
        Layer 3
        ...
        Layer L



═══════════════════════════════════════════════════════════════════════════



                    ITERATIVE REFINEMENT

Pass 1
   │
   ▼
Coarse Representation

   │
   ▼

Pass 2
   │
   ▼
Refined Representation

   │
   ▼

Pass 3
   │
   ▼
Better Representation

   │
   ▼

Pass 4
   │
   ▼
Stable Representation


Interpretation:

h⁽ᵗ⁺¹⁾ = F(h⁽ᵗ⁾)



The same layer progressively improves
the hidden representation.



═══════════════════════════════════════════════════════════════════════════



                     LAYER RECURRENCE

Instead of

A → B → C

Use

A → A → A → A
        │
        ▼
B → B → B → B
        │
        ▼
C → C → C → C


Example in x-transformers:

(A,F)
(A,F)
(A,F)
(A,F)

(B,F)
(B,F)
(B,F)
(B,F)

(C,F)
(C,F)
(C,F)
(C,F)



═══════════════════════════════════════════════════════════════════════════



               CONNECTION TO MODERN TRANSFORMERS


Transformer (2017)
        │
        ▼

ALBERT (2019)
Cross-Layer Sharing
        │
        ▼

Weight-Tied Layers
        │
        ▼

Universal Transformer
        │
        ▼

Layer Recurrence
        │
        ▼

Recurrent Transformer
        │
        ▼

Deep Equilibrium Models
        │
        ▼

x-transformers


═══════════════════════════════════════════════════════════════════════════



                     ADVANTAGES

✓ Parameter Reduction

      O(LP)
         ↓
       O(P)

✓ Better Memory Efficiency

✓ Natural Regularization

✓ Iterative Reasoning

✓ Test-Time Scaling

✓ Deep Effective Computation



═══════════════════════════════════════════════════════════════════════════



                     LIMITATIONS

✗ Less Layer Specialization

✗ Possible Fixed-Point Collapse

✗ Reduced Representation Diversity

✗ Harder Optimization for Very Deep Recurrence



═══════════════════════════════════════════════════════════════════════════



                    KEY TAKEAWAY


Weight-Tied Layers transforms

      F₁ → F₂ → F₃ → ... → Fᴸ

into

      F → F → F → ... → F


or mathematically


      hₗ₊₁ = F(hₗ ; θ)


making Transformer behave as an
iterative refinement system rather
than a stack of independent layers.


This idea forms the conceptual bridge:

ALBERT
   ↓
Weight Sharing
   ↓
Universal Transformer
   ↓
Layer Recurrence
   ↓
x-transformers
   ↓
Deep Equilibrium Models
```
