# Intra-attention Gating on Values — Overview Diagram

> Một sơ đồ tổng quát của kiến trúc **Intra-attention Gating on Values** trong AlphaFold2 và x-transformers.

---

# 1. Big Picture

```text

                    INTRA-ATTENTION GATING ON VALUES

                    "Control What To Write Back"

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                     INPUT REPRESENTATION X                          │
│                                                                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        │
         ┌──────────────┴──────────────┐
         │                             │
         │                             │
         ▼                             ▼

┌───────────────────┐      ┌────────────────────────┐
│ Attention Branch  │      │      Gate Branch       │
└─────────┬─────────┘      └────────────┬───────────┘
          │                             │
          │                             │
          ▼                             ▼

    Q = XWQ                      G = σ(XWG)

    K = XWK

    V = XWV

          │                             │
          ▼                             │

  Softmax(QKᵀ/√d)

          │                             │
          ▼                             │

       O = AV                           │

          │                             │
          └──────────────┬──────────────┘
                         │

                         ▼

                  O_gate = G ⊙ O

                         │

                         ▼

                    Linear WO

                         │

                         ▼

                 Residual Update

                         │

                         ▼

                  Transformer Output

```

---

# 2. Attention chuẩn vs Gated Attention

```text

STANDARD ATTENTION

Input
  │
  ▼

Attention

  │
  ▼

AV

  │
  ▼

Output Projection

  │
  ▼

Residual Stream



────────────────────────────────────



INTRA-ATTENTION GATING

Input
  │
  ├────────────► Gate
  │                  │
  ▼                  ▼

Attention          Sigmoid

  │                  │
  ▼                  ▼

AV                G

  │                  │
  └──────────┬───────┘
             ▼

         G ⊙ AV

             │
             ▼

      Output Projection

             │
             ▼

       Residual Stream

```

---

# 3. Luồng toán học

```text

Input X
   │
   ▼

Query  Key  Value

   │
   ▼

Attention Scores

      QKᵀ
──────────────
    √d

   │
   ▼

Softmax

   │
   ▼

Attention Map A

   │
   ▼

Aggregated Values

O = AV

   │
   │
   ├─────────────────────────┐
   │                         │
   ▼                         ▼

Gate Projection        G = σ(XWG)

   │                         │
   └──────────────┬──────────┘
                  ▼

          O_gate = G ⊙ O

                  │
                  ▼

           Output Projection

                  │
                  ▼

              Final Output

```

---

# 4. Ý nghĩa Information Flow

```text

Standard Transformer

           Read
             │
             ▼

       Attention

             │
             ▼

            Write



─────────────────────────────────



Gated Transformer

           Read
             │
             ▼

       Attention

             │
             ▼

      Aggregated Value

             │

             ▼

       Write Controller
        (Input Gate)

             │
             ▼

            Write

```

Attention chuẩn chỉ học:

```text
Where to Read ?
```

Intra-attention Gating học thêm:

```text
How Much to Write ?
```

---

# 5. Góc nhìn Residual Update

Transformer thực chất học:

```math
x_{l+1} = x_l + \Delta x_l
```

Attention chuẩn:

```math
\Delta x_l = AVW_O
```

Gated Attention:

```math
\Delta x_l = (G \odot AV)W_O
```

Minh họa:

```text

Residual Stream

x_l
 │
 │
 ▼

+ Δx

 │

 ▼

x_(l+1)



Attention:
Δx = AVW_O


Gated Attention:
Δx = (G ⊙ AV)W_O


Gate quyết định:

0  → Không ghi

0.5 → Ghi một phần

1  → Ghi toàn bộ

```

---

# 6. Quan hệ với AlphaFold2

```text

                    AlphaFold2 Style Attention


                    Pair/MSA Representation
                               │
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼

          Attention                    Gate Projection

               │                               │
               ▼                               ▼

      Aggregated Values                 Sigmoid Gate

               │                               │
               └───────────────┬───────────────┘
                               ▼

                        Element-wise

                          Multiply

                               │
                               ▼

                           Output

```

Triết lý của AlphaFold2:

```text
Read Information
       ↓
Filter Information
       ↓
Write Information
```

thay vì:

```text
Read Information
       ↓
Write Information
```

---

# 7. So sánh với các kiến trúc liên quan

```text

                    Attention Variants

                           Attention
                               │
        ┌──────────────┬───────┴──────────────┐
        │              │                      │
        ▼              ▼                      ▼

     AoA         Talking Heads       Gate Values

        │              │                      │

 Extra Attention    Mix Heads       Control Update

        │              │                      │

 Better Read     Better Routing     Better Writing

```

| Kiến trúc | Ý tưởng chính |
|------------|---------------|
| Multi-Head Attention | Học nơi cần đọc |
| Talking-Heads | Trộn thông tin giữa các head |
| Attention-on-Attention | Attention trên attention |
| Memory Tokens | Tạo bộ nhớ toàn cục |
| **Gate Values** | Kiểm soát lượng thông tin ghi trở lại |

---

# 8. Sơ đồ tổng kết một trang

```text

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│            INTRA-ATTENTION GATING ON VALUES                 │
│                                                             │
│      Inspired by AlphaFold2 and adopted in x-transformers   │
│                                                             │
└─────────────────────────────────────────────────────────────┘


                 INPUT X
                     │
      ┌──────────────┴──────────────┐
      │                             │
      ▼                             ▼

 Multi-Head Attention        Gate Generator

 Q,K,V                       G = σ(XWG)

      │                             │
      ▼                             │

 O = AV                             │

      └──────────────┬──────────────┘
                     ▼

              O_gate = G ⊙ O

                     │

                     ▼

               Output Linear

                     │

                     ▼

              Residual Update

                     │

                     ▼

                 Transformer


──────────────────────────────────────────────

Core Equation

O = AV

G = σ(XW_G)

O_gate = G ⊙ O

Y = O_gate W_O


──────────────────────────────────────────────

Attention decides:

    "Where to Read"

Gate decides:

    "How Much to Write"

──────────────────────────────────────────────

Benefits

✓ Better update control

✓ Noise suppression

✓ Stronger residual regulation

✓ Very low computational overhead

✓ Effective in AlphaFold2

✓ Available in x-transformers

```

---

## Hình gợi ý chèn vào README

```html
<img src="assets/intra_attention_gating_overview.png" width="100%">
```

Nên đặt sơ đồ này ngay sau phần **Introduction** của README, trước phần **Mathematical Formulation**, vì nó tóm tắt toàn bộ kiến trúc từ góc nhìn Attention → Gate → Residual Update trong một trang duy nhất.