## Transformer-XL Architecture Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                    TRANSFORMER-XL                            │
│                                                              │
│      Transformer + Memory + Recurrence + Relative Pos        │
└──────────────────────────────────────────────────────────────┘


                 LONG SEQUENCE

x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 ...
│──────────────│──────────────│──────────────│
   Segment 1       Segment 2       Segment 3


=================================================================
1. VANILLA TRANSFORMER
=================================================================

 Segment 1              Segment 2

┌──────────────┐      ┌──────────────┐
│ x1 x2 x3 x4  │      │ x5 x6 x7 x8  │
└──────┬───────┘      └──────┬───────┘
       ▼                     ▼

   Self-Attention       Self-Attention

       ▼                     ▼

 Hidden States        Hidden States

       ✗
       │
       │  No information flow
       ▼

 Segment boundary breaks context


Effective Context:

┌──────────────┐
│ Current Only │
└──────────────┘



=================================================================
2. TRANSFORMER-XL
=================================================================

 Segment 1

┌──────────────┐
│ x1 x2 x3 x4  │
└──────┬───────┘
       ▼

 Self-Attention

       ▼

 Hidden States
 h1 h2 h3 h4

       │
       ▼

 Memory Cache
 M1


                    ▼ reused ▼


 Segment 2

           Memory
┌───────────────────────┐
│ h1 h2 h3 h4           │
└──────────┬────────────┘
           │

           ▼

┌──────────────┐
│ x5 x6 x7 x8  │
└──────┬───────┘
       ▼

Self-Attention

Q = Current Segment

K,V = [Memory ; Current]

       ▼

Hidden States

       ▼

Memory Cache
M2


                    ▼ reused ▼


 Segment 3

Uses:

[M2 ; Current Segment]



=================================================================
3. MEMORY AUGMENTED ATTENTION
=================================================================

                Query

                  Q
                  │
                  ▼

         ┌─────────────────┐
         │    Attention    │
         └─────────────────┘

             ▲         ▲

             │         │

             │         │

        Current      Memory
        Segment      Cache

             │         │

             ▼         ▼

          K,V      K,V


K = [M ; H]WK

V = [M ; H]WV


where

M = previous segment memory

H = current hidden states



=================================================================
4. SEGMENT-LEVEL RECURRENCE
=================================================================

Layer l

Segment n-1

      H(n-1,l)

           │
           │ Stop Gradient
           ▼

      Memory M(n,l)

           │
           ▼

Segment n

      Attention

           ▲

           │

      Reused Memory


Mathematically

M_n^l = SG(H_(n-1)^l)



=================================================================
5. RELATIVE POSITIONAL ENCODING
=================================================================

Vanilla Transformer

Position:

p1 p2 p3 p4 p5 ...


Attention:

A(i,j)

depends on

(i , j)


Problem:

Memory reuse creates
position ambiguity.


------------------------------------------------------


Transformer-XL

Uses:

Relative Distance

r(i-j)


Example

Token i

     ◄───── 3 ─────►

Token j


Attention Score

A(i,j)

depends on

(i-j)

instead of

(i , j)


Benefits

✓ Translation Invariance

✓ Segment Reuse

✓ Long Context Stability



=================================================================
6. COMPLETE ATTENTION SCORE
=================================================================

                Content
                    │
                    ▼

                qᵀk

                    │

                    ▼

      ┌────────────────────────────┐
      │ Transformer-XL Attention   │
      └────────────────────────────┘

                    ▲

                    │

        ┌───────────┼───────────┐

        ▼           ▼           ▼

      qᵀr         uᵀk         vᵀr

 Relative     Global      Global
 Position     Content     Position


Final Score


A_ij = qᵢᵀkⱼ + qᵢᵀr(i-j) + uᵀkⱼ + vᵀr(i-j)



=================================================================
7. EVOLUTION TO MODERN LLMs
=================================================================

Transformer (2017)
         │
         ▼

Transformer-XL (2019)

         │
         ├── Memory
         │
         ├── Recurrence
         │
         └── Relative Position

         ▼

Longformer

         ▼

BigBird

         ▼

Compressive Transformer

         ▼

RETRO

         ▼

GPT KV Cache

         ▼

Long Context LLMs

         ▼

Modern x-Transformers



=================================================================
KEY INSIGHTS
=================================================================

1. Memory is reused across segments

       H(n-1)
          │
          ▼
         M(n)

2. Context extends beyond segment length

       Context = L + M

3. Relative positions replace absolute positions

       r(i-j)

4. Transformer gains recurrence
   without becoming an RNN

5. Foundation of modern KV Cache
   and long-context language models
```
