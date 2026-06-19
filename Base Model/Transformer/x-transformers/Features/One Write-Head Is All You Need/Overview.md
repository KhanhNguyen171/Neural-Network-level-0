# One Write-Head Is All You Need — Architecture Overview

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                    ONE WRITE-HEAD IS ALL YOU NEED                         │
│                 Fast Transformer Decoding with MQA                        │
└───────────────────────────────────────────────────────────────────────────┘

Problem = Autoregressive Decoding

x1 → x2 → x3 → x4 → ... → xt

At every decoding step:

      Query
         │
         ▼
   Attention Layer
         │
         ▼
      Next Token

The bottleneck is NOT computation.

The bottleneck is:

      KV Cache Memory Bandwidth

because K and V from ALL previous tokens
must be loaded repeatedly during generation.



──────────────────────────────────────────────────────────────────────────────
1. STANDARD MULTI-HEAD ATTENTION (MHA)
──────────────────────────────────────────────────────────────────────────────

                    Head 1
              Q1 ───► K1,V1

                    Head 2
              Q2 ───► K2,V2

                    Head 3
              Q3 ───► K3,V3

                    ...
              
                    Head H
              QH ───► KH,VH


KV Cache

Token 1 : [K1,V1][K2,V2]...[KH,VH]
Token 2 : [K1,V1][K2,V2]...[KH,VH]
Token 3 : [K1,V1][K2,V2]...[KH,VH]
   ...
Token N : [K1,V1][K2,V2]...[KH,VH]

Memory Complexity

O(HN)

where

H = number of heads
N = sequence length


Problem:

More heads
      ↓
More KV cache
      ↓
More memory traffic
      ↓
Slower inference



──────────────────────────────────────────────────────────────────────────────
2. CORE IDEA OF THE PAPER
──────────────────────────────────────────────────────────────────────────────

Observation:

Queries need multiple heads.

Keys and Values may not.

Therefore:

Keep many Query heads

BUT

Share a single KV head.


          MANY READ HEADS
                 │
                 ▼

      Q1
      Q2
      Q3
      ...
      QH

                 │
                 ▼

        ONE SHARED KV HEAD

                 K
                 V

This is why the paper is called:

      "One Write-Head Is All You Need"



──────────────────────────────────────────────────────────────────────────────
3. MULTI-QUERY ATTENTION (MQA)
──────────────────────────────────────────────────────────────────────────────

                    Q1
                     │
                     ▼
                  K,V

                    Q2
                     │
                     ▼
                  K,V

                    Q3
                     │
                     ▼
                  K,V

                    ...
                     │
                     ▼

                    QH
                     │
                     ▼
                  K,V


All Query Heads
share the same Key and Value projections.



Mathematically

Qi = XWiQ

K = XWK

V = XWV


Attention

Ai = softmax(QiKT / √dh)

Outputi = AiV



──────────────────────────────────────────────────────────────────────────────
4. KV CACHE COMPARISON
──────────────────────────────────────────────────────────────────────────────

Multi-Head Attention

Token 1 : [KV1][KV2][KV3][KV4][KV5][KV6][KV7][KV8]
Token 2 : [KV1][KV2][KV3][KV4][KV5][KV6][KV7][KV8]
Token 3 : [KV1][KV2][KV3][KV4][KV5][KV6][KV7][KV8]

Memory

O(HN)



Multi-Query Attention

Token 1 : [KV]
Token 2 : [KV]
Token 3 : [KV]

Memory

O(N)



Memory Reduction ≈ H × smaller

Example

H = 16

KV Cache

16x smaller



──────────────────────────────────────────────────────────────────────────────
5. WHY INFERENCE BECOMES FASTER
──────────────────────────────────────────────────────────────────────────────

MHA

GPU
 │
 ├── Load KV Head 1
 ├── Load KV Head 2
 ├── Load KV Head 3
 ├── ...
 └── Load KV Head H

Huge memory traffic



MQA

GPU
 │
 └── Load one KV cache

Small memory traffic



Result

Less HBM Access
        ↓
Less Memory Bandwidth Pressure
        ↓
Higher Throughput
        ↓
Faster Decoding



──────────────────────────────────────────────────────────────────────────────
6. EVOLUTION OF ATTENTION ARCHITECTURES
──────────────────────────────────────────────────────────────────────────────

Attention Is All You Need (2017)

                │
                ▼

      Multi-Head Attention
          (MHA)

                │
                ▼

 One Write-Head Is All You Need
          (2019)

                │
                ▼

     Multi-Query Attention
           (MQA)

                │
                ▼

 AlphaCode
 PaLM

                │
                ▼

 Grouped Query Attention
          (GQA)

                │
                ▼

 LLaMA-2
 Gemini
 Gemma
 Mistral
 Mixtral



──────────────────────────────────────────────────────────────────────────────
7. MHA → MQA → GQA
──────────────────────────────────────────────────────────────────────────────

MHA

Q1 → KV1
Q2 → KV2
Q3 → KV3
Q4 → KV4

Maximum Quality
Maximum Memory



MQA

Q1 → KV
Q2 → KV
Q3 → KV
Q4 → KV

Minimum Memory
Maximum Speed



GQA

Q1 → KV1
Q2 → KV1

Q3 → KV2
Q4 → KV2

Quality ≈ MHA
Memory ≈ MQA

Best Trade-off



──────────────────────────────────────────────────────────────────────────────
FINAL TAKEAWAY
──────────────────────────────────────────────────────────────────────────────

The paper discovered that:

    Multiple Query Heads are necessary

but

    Multiple Key/Value Heads are not.

Therefore:

      MHA
        ↓
   Share K,V
        ↓
      MQA
        ↓
 Huge KV Cache Reduction
        ↓
 Faster Autoregressive Decoding
        ↓
 Foundation of Modern LLM Inference
```
