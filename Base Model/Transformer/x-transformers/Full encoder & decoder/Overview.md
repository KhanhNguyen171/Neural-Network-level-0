## Full Encoder-Decoder Transformer Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        INPUT TOKENS                         │
│                                                             │
│   x1      x2      x3      ...      xn                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Token Embedding    │
                 └──────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Position Encoding    │
                 └──────────────────────┘
                            │
                            ▼

╔══════════════════════════════════════════════════════════════╗
║                        ENCODER STACK                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ Encoder Layer 1                                      │    ║
║  │                                                      │    ║
║  │ Self Attention                                       │    ║
║  │        ↓                                             │    ║
║  │ Feed Forward Network                                 │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║                         ⋮                                    ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ Encoder Layer L                                      │    ║ 
║  │                                                      │    ║
║  │ Self Attention                                       │    ║
║  │        ↓                                             │    ║
║  │ Feed Forward Network                                 │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
                            │
                            ▼

┌─────────────────────────────────────────────────────────────┐
│                     ENCODER MEMORY                          │
│                                                             │
│ M = {m₁,m₂,m₃,...,mₙ}                                       │
│                                                             │
│ Contextual Representations                                  │
│ Global Semantic Memory                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │
                            │ K,V
                            │
                            ▼

══════════════════════════════════════════════════════════════════════

                 AUTOREGRESSIVE GENERATION

══════════════════════════════════════════════════════════════════════

                            ▲
                            │
                            │ Previous Tokens
                            │
┌─────────────────────────────────────────────────────────────┐
│               BOS, y₁, y₂, ..., yₜ₋₁                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼

╔══════════════════════════════════════════════════════════════╗
║                        DECODER STACK                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ Decoder Layer 1                                      │    ║
║  │                                                      │    ║
║  │ Masked Self Attention                                │    ║
║  │            ↓                                         │    ║
║  │ Cross Attention  ◄──────── Encoder Memory            │    ║
║  │            ↓                                         │    ║
║  │ Feed Forward Network                                 │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║                          ⋮                                   ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ Decoder Layer L                                      │    ║
║  │                                                      │    ║
║  │ Masked Self Attention                                │    ║
║  │            ↓                                         │    ║
║  │ Cross Attention  ◄──────── Encoder Memory            │    ║
║  │            ↓                                         │    ║
║  │ Feed Forward Network                                 │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
                            │
                            ▼

                 ┌──────────────────────┐
                 │ Linear Projection    │
                 └──────────────────────┘
                            │
                            ▼

                 ┌──────────────────────┐
                 │ Softmax Vocabulary   │
                 └──────────────────────┘
                            │
                            ▼

                 ┌──────────────────────┐
                 │ Next Token yₜ        │
                 └──────────────────────┘
```

---

## Information Flow

```text
Input Sequence
      │
      ▼
Encoder
      │
      ▼
Context Memory
      │
      ▼
Cross Attention
      │
      ▼
Decoder
      │
      ▼
Output Sequence
```

---

## Attention Flow

```text
ENCODER

      Q
      │
      ▼
Q × Kᵀ
      ▲
      │
      K

      │
      ▼
 Softmax
      │
      ▼
Weighted V

(All tokens can attend to all tokens)



DECODER

      Q
      │
      ▼
Q × Kᵀ + Causal Mask
      ▲
      │
      K

      │
      ▼
 Softmax
      │
      ▼
Weighted V

(Only attend to previous tokens)



CROSS ATTENTION

 Decoder Hidden States
          │
          ▼
          Q

          │
          ▼

Encoder Memory ──► K
Encoder Memory ──► V

          │
          ▼

      Cross Context

          │
          ▼

Decoder Update
```

---

## Mathematical View

```text
Input X
   │
   ▼

M = Encoder(X)

   │
   ▼

Hₜ = Decoder(y<t , M)

   │
   ▼

zₜ = WHₜ + b

   │
   ▼

P(yₜ|y<t,X)

   │
   ▼

Generate Token
```

---

## Relationship with Modern Transformer Models

```text
                     Transformer Family

                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼

 Encoder Only        Encoder-Decoder       Decoder Only

     BERT             T5 / UL2              GPT
                       PaLI
                     Flamingo
                      Kosmos

        │                    │                    │
        ▼                    ▼                    ▼

 Understanding      Seq2Seq Learning     Language Modeling

                            │
                            ▼

                 Multimodal Foundation Models
```

---

## Core Components

```text
Input Tokens
      │
      ▼
Embedding
      │
      ▼
Positional Encoding
      │
      ▼
Encoder Self Attention
      │
      ▼
Encoder Memory
      │
      ▼
Cross Attention
      │
      ▼
Masked Self Attention
      │
      ▼
Feed Forward Network
      │
      ▼
Vocabulary Projection
      │
      ▼
Generated Tokens
```
