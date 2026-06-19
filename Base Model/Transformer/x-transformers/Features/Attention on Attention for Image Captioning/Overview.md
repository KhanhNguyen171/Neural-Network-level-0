# Attention on Attention (AoA) - Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ATTENTION ON ATTENTION (AoA)                           │
│               Attention on Attention for Image Captioning                  │
│                          Huang et al., ICCV 2019                           │
└─────────────────────────────────────────────────────────────────────────────┘


                    STANDARD ATTENTION

        Query(Q)            Key(K)            Value(V)
            │                  │                  │
            └──────────┬───────┴───────┬──────────┘
                       │               │
                       ▼               ▼

                 Similarity Scores

                    QKᵀ / √d

                         │
                         ▼

                      Softmax

                         │
                         ▼

                  Attention Map

                         │
                         ▼

                    Weighted Sum

                         │
                         ▼

                   Context Vector

                         V̂

                         │
                         │
                         ▼

═══════════════════════════════════════════════════════════════════════════════

                  ATTENTION ON ATTENTION

                         V̂
                         │
                         │
                         ▼

                    Concatenate

                     [V̂ ; Q]

                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼

  Information Branch             Gate Branch

      Linear                      Linear
          │                           │
          ▼                           ▼

          I                       Sigmoid

                                      │
                                      ▼

                                      G

          └──────────────┬──────────────┘
                         │
                         ▼

                      I ⊙ G

                         │
                         ▼

                  Refined Context

═══════════════════════════════════════════════════════════════════════════════

              CORE IDEA OF THE PAPER

          Traditional Attention

          "Where should I look?"

                         │
                         ▼

              Context Retrieval

                         │
                         ▼

───────────────────────────────────────────────────────────────────────────────

           Attention on Attention

          "What should I keep?"

                         │
                         ▼

              Feature Selection

                         │
                         ▼

             Information Filtering

═══════════════════════════════════════════════════════════════════════════════

                   IMAGE ENCODER

             Region Features (CNN)

                         │
                         ▼

              Multi-Head Attention

                         │
                         ▼

                        AoA

                         │
                         ▼

               Refined Region Features

                         │
                         ▼

              Better Object Relations

═══════════════════════════════════════════════════════════════════════════════

                   CAPTION DECODER

                  Previous Words

                         │
                         ▼

                        LSTM

                         │
                         ▼

                 Visual Attention

                         │
                         ▼

                        AoA

                         │
                         ▼

                 Word Prediction

                         │
                         ▼

              Next Caption Token

═══════════════════════════════════════════════════════════════════════════════

               RELATION TO MODERN LLMS

              Attention Output
                      │
                      ▼

               Output Gating

                      │
                      ▼

         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼

        GLU         GEGLU       SwiGLU

         │            │            │
         └────────────┼────────────┘
                      │
                      ▼

              Modern Transformers

                  PaLM
                  LLaMA
              x-transformers
            Gated Attention Units

═══════════════════════════════════════════════════════════════════════════════

                  MATHEMATICAL FORM

        Attention(Q,K,V)

             = Softmax(QKᵀ/√d)V

                        │
                        ▼

                       V̂

                        │

        I = W_I[V̂ ; Q] + b_I

        G = σ(W_G[V̂ ; Q] + b_G)

                        │

        AoA = I ⊙ G

═══════════════════════════════════════════════════════════════════════════════

                     MAIN CONTRIBUTION

      Attention retrieves information.

      AoA evaluates information.

      Attention decides:

             "WHERE TO ATTEND"

      AoA decides:

             "WHAT TO USE"

═══════════════════════════════════════════════════════════════════════════════
```

### Kiến thức cần nhớ sau khi đọc paper

```text
Attention
     │
     ▼
Context Retrieval
     │
     ▼
AoA
     │
     ▼
Feature Gating
     │
     ▼
Refined Context
     │
     ▼
Decoder / Transformer Layer
```

### Một câu tóm tắt

```text
AoA = Attention + Information Selection

Attention:
    tìm thông tin liên quan

AoA:
    đánh giá và lọc thông tin vừa tìm được

=> tiền thân của các kiến trúc Gated Transformer hiện đại.
```
