```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY TOKENS ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘


INPUT SEQUENCE

x₁      x₂      x₃      x₄      x₅      x₆      x₇      x₈
│       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┘

                              │
                              ▼

        CONCATENATE LEARNABLE MEMORY TOKENS

┌──────┐ ┌──────┐ ┌──────┐
│  m₁  │ │  m₂  │ │  m₃  │
└──────┘ └──────┘ └──────┘

                              +

[x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈]

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELF-ATTENTION LAYER                               │
└─────────────────────────────────────────────────────────────────────────────┘


         WRITE INFORMATION TO MEMORY

               x₁ ───────────────┐
               x₂ ────────────┐  │
               x₃ ────────┐   │  │
               x₄ ────┐   │   │  │
                       ▼   ▼   ▼  ▼

                 ┌───────────────┐
                 │ Memory Tokens │
                 │               │
                 │  m₁  m₂  m₃   │
                 └───────────────┘

                       ▲   ▲   ▲
               x₅ ────┘   │   │
               x₆ ────────┘   │
               x₇ ────────────┘
               x₈ ─────────────┘


───────────────────────────────────────────────────────────────────────────────


          READ INFORMATION FROM MEMORY

                 ┌───────────────┐
                 │ Memory Tokens │
                 │               │
                 │  m₁  m₂  m₃   │
                 └───────────────┘

                    │   │   │
        ┌───────────┴───┴───┴───────────┐
        ▼       ▼      ▼       ▼        ▼

       x₁      x₂     x₃      x₄      x₅ ...



───────────────────────────────────────────────────────────────────────────────


INFORMATION FLOW

                xᵢ
                 │
                 ▼

              Memory

                 │
                 ▼

                xⱼ


Traditional Transformer:

xᵢ ─────────────────────────────► xⱼ

Memory Transformer:

xᵢ ─────► Memory ─────► xⱼ



───────────────────────────────────────────────────────────────────────────────


MEMORY AS GLOBAL BOTTLENECK

        Large Sequence

┌──────────────────────────────┐
│ x₁ x₂ x₃ ... x₄₀₉₆           │
└──────────────────────────────┘

               │
               ▼

       ┌────────────────┐
       │ Memory Tokens  │
       │ m₁ ... m₃₂     │
       └────────────────┘

               │
               ▼

      Global Representation


Compression:

R^(4096 × d)
        ↓
R^(32 × d)



───────────────────────────────────────────────────────────────────────────────


MULTI-LAYER MEMORY EVOLUTION

Layer 1

[m₁ m₂ m₃]
      │
      ▼

Layer 2

[m₁' m₂' m₃']
      │
      ▼

Layer 3

[m₁'' m₂'' m₃'']
      │
      ▼

Layer L

[M_global]


Memory accumulates information
across Transformer layers.



───────────────────────────────────────────────────────────────────────────────


RECURRENT MEMORY EXTENSION

Segment 1

[M₀] + X₁
      │
      ▼
[M₁]

      │

Segment 2

[M₁] + X₂
      │
      ▼
[M₂]

      │

Segment 3

[M₂] + X₃
      │
      ▼
[M₃]



Memory persists beyond context window.



───────────────────────────────────────────────────────────────────────────────


EVOLUTION OF THE IDEA


CLS Token
    │
    ▼

Memory Tokens
    │
    ▼

Perceiver Latents
    │
    ▼

Recurrent Memory Transformer
    │
    ▼

Long-Context Transformer
    │
    ▼

Modern LLM Memory Systems
```
