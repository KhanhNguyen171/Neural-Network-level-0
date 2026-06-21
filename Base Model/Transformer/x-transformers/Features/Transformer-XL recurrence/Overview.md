# Transformer-XL Recurrence Overview

> Hình minh họa tổng quát toàn bộ kiến trúc và cơ chế hoạt động của **Transformer-XL Recurrence Mechanism**.

---

# 1. Từ Transformer chuẩn đến Transformer-XL

```text
Standard Transformer

Context Window = L

┌───────────────────────────────┐
│ x1 x2 x3 ... xL               │
└───────────────────────────────┘

Token xL chỉ nhìn thấy:

x1 ... xL

────────────────────────────────────────────

Nếu chuỗi dài hơn:

x1 x2 ... xL | xL+1 ...

=> toàn bộ thông tin trước đó bị mất.
```

---

```text
Transformer-XL

Effective Context = L + M

Memory:
m1 m2 m3 ... mM

Current Segment:
x1 x2 x3 ... xL

┌────────────────────────────────────┐
│ m1 ... mM x1 ... xL                │
└────────────────────────────────────┘

Token hiện tại có thể attention tới:

Past Memory + Current Segment
```

---

# 2. Segment-Level Recurrence

```text
Long Sequence

x1 x2 x3 ............................................... x10000

      │
      ▼

┌──────────┐
│Segment 1 │
└──────────┘
      │
      ▼
  Hidden States
      │
      ▼
   Memory M1
      │
      ▼
┌──────────┐
│Segment 2 │
└──────────┘
      │
      ▼
   Memory M2
      │
      ▼
┌──────────┐
│Segment 3 │
└──────────┘
      │
      ▼
      ...
```

---

# 3. Recurrence Mechanism

```text
Segment t-1

┌────────────────────────────┐
│ h1 h2 h3 ... hL            │
└────────────────────────────┘
               │
               │ detach()
               ▼
         Memory Bank Mt-1
               │
               │
               ▼
Segment t

┌────────────────────────────┐
│ x1 x2 x3 ... xL            │
└────────────────────────────┘
```

Memory được tái sử dụng:

```math
M_t = SG(H_t)
```

với:

```math
SG(\cdot)
```

là:

```python
detach()
```

---

# 4. Attention Architecture

```text
                 Query (Current Segment)
                            │
                            ▼
                  ┌────────────────┐
                  │      Q         │
                  └────────────────┘

                            │
                            ▼

      ┌─────────────────────────────────────┐
      │        Memory + Current             │
      │                                     │
      │  m1 ... mM x1 ... xL               │
      └─────────────────────────────────────┘
                    │             │
                    ▼             ▼
                    K             V

                            │
                            ▼

                  Scaled Dot Product
                            │
                            ▼
                       Attention
                            │
                            ▼
                          Output
```

---

# 5. Effective Context Expansion

```text
Standard Transformer

┌────────────────────┐
│ Context = L        │
└────────────────────┘


Transformer-XL

┌────────────────────┐
│ Memory = M         │
├────────────────────┤
│ Current = L        │
└────────────────────┘

Effective Context:

L + M
```

---

# 6. Information Flow

```text
Forward Pass

Segment1 ─────► Segment2 ─────► Segment3
     │               │               │
     ▼               ▼               ▼
  Memory1         Memory2         Memory3


Backward Pass

Segment3 ◄──── gradients
Segment2 ◄──── gradients
Segment1 ◄──── gradients

Memory does NOT receive gradients.
```

```text
detach(memory)
```

ngăn computational graph phát triển vô hạn.

---

# 7. Relative Positional Encoding

Transformer chuẩn:

```text
Position:
1 2 3 ... 512
```

Segment tiếp theo:

```text
Position:
1 2 3 ... 512
```

→ xung đột vị trí.

---

Transformer-XL:

```text
Distance Based Encoding

x_i <────── i-j ──────> x_j
```

Attention score:

```math
A_{i,j} = q_i^T k_j + q_i^T r_{i-j} + u^T k_j + v^T r_{i-j}
```

Giúp memory có thể tái sử dụng giữa các segment.

---

# 8. Training Pipeline

```text
Long Sequence
─────────────────────────────────────

Segment1
     │
     ▼
 Transformer
     │
     ▼
 Memory1
     │
     ▼
Segment2
     │
     ▼
 Transformer
     │
     ▼
 Memory2
     │
     ▼
Segment3
     │
     ▼
 Transformer
     │
     ▼
 Memory3
```

---

# 9. Inference Pipeline

```text
Prompt
   │
   ▼
Segment1
   │
   ▼
Memory1
   │
   ▼
Segment2
   │
   ▼
Memory2
   │
   ▼
Generate Next Tokens
```

Không cần recompute toàn bộ lịch sử.

---

# 10. Complexity

```text
Standard Transformer

O(L²)
```

```text
Transformer-XL

O(L(L+M))
```

với:

```math
M \ll T
```

nên chi phí thấp hơn attention trên toàn bộ chuỗi.

---

# 11. x-transformers Implementation

```python
model_xl = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 512,
    max_mem_len = 2048,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rel_pos_bias = True
    )
)
```

---

```python
logits1, mem1 = model_xl(
    seg1,
    return_mems=True
)

logits2, mem2 = model_xl(
    seg2,
    mems=mem1,
    return_mems=True
)

logits3, mem3 = model_xl(
    seg3,
    mems=mem2,
    return_mems=True
)
```

---

# 12. Tổng quan kiến trúc

```text
                        Transformer-XL
──────────────────────────────────────────────────────────

                  ┌──────────────────┐
                  │ Relative Pos Enc │
                  └──────────────────┘
                             │
                             ▼
┌──────────┐       ┌─────────────────────┐
│ Memory   │──────►│ Multi-Head Attention│
└──────────┘       └─────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ Feed Forward     │
                  └──────────────────┘
                             │
                             ▼
                      Hidden States
                             │
                             ▼
                        detach()
                             │
                             ▼
                        New Memory
```

---

# Key Takeaways

```text
✓ Segment-Level Recurrence

✓ Effective Context = L + M

✓ Reuse Hidden States as Memory

✓ Relative Positional Encoding

✓ No Gradient Through Memory

✓ Efficient Long-Range Dependency Modeling

✓ Foundation of Modern Long-Context Transformers
```
