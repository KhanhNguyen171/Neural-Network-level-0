## Talking-Heads Attention Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                   TALKING-HEADS ATTENTION                    │
│             (Noam Shazeer, 2020, arXiv:2003.02436)           │
└───────────────────────────────────────────────────────────────┘


                     Input Tokens
                            │
                            ▼
                    X ∈ R^(n×dX)
                            │
                            ▼
          ┌────────────────────────────────┐
          │ Linear Query Projection (Pq)   │
          └────────────────────────────────┘
                            │
                            ▼
                  Q ∈ R^(n×dk×hk)

                            ▲
                            │

          ┌────────────────────────────────┐
          │ Linear Key Projection (Pk)     │
          └────────────────────────────────┘
                            │
                            ▼
                  K ∈ R^(m×dk×hk)


          ┌────────────────────────────────┐
          │ Linear Value Projection (Pv)   │
          └────────────────────────────────┘
                            │
                            ▼
                  V ∈ R^(m×dv×hv)



══════════════════════════════════════════════════════════════════
                    STANDARD MULTI-HEAD
══════════════════════════════════════════════════════════════════

        Head1 ──────────────┐
        Head2 ──────────────┤
        Head3 ──────────────┤
        Head4 ──────────────┘

              Independent Computation

                    QKᵀ
                     │
                     ▼

                  Softmax

                     │
                     ▼

                    AV

                     │
                     ▼

                   Output



══════════════════════════════════════════════════════════════════
                   TALKING-HEADS ATTENTION
══════════════════════════════════════════════════════════════════


                  Q × Kᵀ

                     │

                     ▼

       J ∈ R^(n×m×hk)   (Raw Attention Logits)

                     │
                     │
                     ▼

      ┌──────────────────────────────┐
      │      LOGIT MIXING (Pl)       │
      │                              │
      │      hk  ─────►  h           │
      └──────────────────────────────┘

                     │

                     ▼

       L ∈ R^(n×m×h)

                     │

                     ▼

                 Softmax

                     │

                     ▼

       W ∈ R^(n×m×h)

                     │
                     │
                     ▼

      ┌──────────────────────────────┐
      │     WEIGHT MIXING (Pw)       │
      │                              │
      │       h ─────► hv            │
      └──────────────────────────────┘

                     │

                     ▼

      U ∈ R^(n×m×hv)

                     │

                     ▼

                  U × V

                     │

                     ▼

      O ∈ R^(n×dv×hv)

                     │

                     ▼

          Output Projection (Po)

                     │

                     ▼

              Y ∈ R^(n×dY)
```

---

## Head Communication Mechanism

```text
STANDARD MULTI-HEAD ATTENTION


 Head1     Head2     Head3     Head4

   │         │         │         │
   ▼         ▼         ▼         ▼

 Independent Independent Independent Independent

   │         │         │         │
   └─────────┴─────────┴─────────┘

                Output


No communication between heads.
```

```text
TALKING-HEADS ATTENTION


            Pre-Softmax Mixing

 Head1 ←──────────────→ Head2
   ↑                      ↓
   │                      │
   ↓                      ↑
 Head3 ←──────────────→ Head4


                    │
                    ▼

                 Softmax


                    │
                    ▼


           Post-Softmax Mixing

 Head1 ←──────────────→ Head2
   ↑                      ↓
   │                      │
   ↓                      ↑
 Head3 ←──────────────→ Head4


                    │
                    ▼

                  Output
```

---

## Tensor Evolution Through The Network

```text
Input

X
│
▼

Queries / Keys / Values

Q : (n × dk × hk)
K : (m × dk × hk)
V : (m × dv × hv)

│
▼

Dot Product

J : (n × m × hk)

│
▼

Logit Projection

Pl : (hk × h)

│
▼

L : (n × m × h)

│
▼

Softmax

W : (n × m × h)

│
▼

Weight Projection

Pw : (h × hv)

│
▼

U : (n × m × hv)

│
▼

Attention × Value

O : (n × dv × hv)

│
▼

Output Projection

Y : (n × dY)
```

---

## Three Distinct Head Spaces

Talking-Heads tổng quát hóa Multi-Head Attention bằng ba không gian head độc lập.

```text
                   hk
        (Query / Key Heads)

                       │

                       ▼

          ┌────────────────────┐
          │      P_l           │
          │  Logit Mixing      │
          └────────────────────┘

                       │

                       ▼

                       h
         (Attention Logit Heads)

                       │

                       ▼

          ┌────────────────────┐
          │      P_w           │
          │ Weight Mixing      │
          └────────────────────┘

                       │

                       ▼

                      hv
          (Value / Output Heads)
```

---

## Core Innovation

```text
Transformer (2017)

    Head_i
       │
       ▼
    Attention_i
       │
       ▼
     Output_i

Independent Heads
────────────────────────────────


Talking-Heads (2020)

 Head_1 ─┐
 Head_2 ─┼──► Mixed Logits
 Head_3 ─┤
 Head_4 ─┘

                ▼

             Softmax

                ▼

 Head_1 ─┐
 Head_2 ─┼──► Mixed Weights
 Head_3 ─┤
 Head_4 ─┘

                ▼

              Output


Collaborative Heads
────────────────────────────────
```

---

### Key Takeaway

```text
Multi-Head Attention

    Head_i ⟂ Head_j


Talking-Heads Attention

    Head_i ↔ Head_j


"Attention heads can communicate
before and after Softmax."
```
