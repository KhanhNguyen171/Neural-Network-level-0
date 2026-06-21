# Overview of T5's Simplified Relative Positional Encoding

---

# 1. Vị trí của T5 Relative Bias trong quá trình phát triển Positional Encoding

```text
Absolute Position Encoding (Transformer 2017)
                     │
                     │ Không mô hình hóa khoảng cách tương đối
                     ▼
Relative Position Representation (Shaw et al., 2018)
                     │
                     │ Chính xác nhưng tốn chi phí tính toán
                     ▼
T5 Relative Position Bias (Raffel et al., 2019)
                     │
                     │ Đơn giản + hiệu quả + chi phí gần như bằng 0
                     ▼
ALiBi ── RoPE ── XPos ── Dynamic Position Bias
```

---

# 2. Ý tưởng cốt lõi

Thay vì:

```math
x_i = e_i + p_i
```

T5 sử dụng:

```math
Attention = Softmax \left( \frac{QK^T}{\sqrt{d_k}} + B \right)V
```

trong đó:

```math
B_{ij} = b(j-i)
```

---

# 3. Kiến trúc tổng quát

```text
                 ┌───────────────────┐
                 │     Input X       │
                 └─────────┬─────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
      Linear             Linear             Linear
        │                  │                  │
        ▼                  ▼                  ▼
        Q                  K                  V
        │                  │
        └──────────┬───────┘
                   │
                   ▼
              QKᵀ / √dk
                   │
                   │
                   │      Relative Distance
                   │      r = j - i
                   │              │
                   │              ▼
                   │      Bucket Mapping
                   │              │
                   │              ▼
                   │      Learnable Bias
                   │              B
                   │              │
                   └───────(+ )───┘
                           │
                           ▼
                        Softmax
                           │
                           ▼
                      Attention Weights
                           │
                           ▼
                           V
                           │
                           ▼
                         Output
```

---

# 4. Relative Distance Matrix

Ví dụ:

```text
Tokens

x1   x2   x3   x4   x5
│    │    │    │    │
▼    ▼    ▼    ▼    ▼

Relative Distance Matrix

      1   2   3   4   5
1     0   1   2   3   4
2    -1   0   1   2   3
3    -2  -1   0   1   2
4    -3  -2  -1   0   1
5    -4  -3  -2  -1   0
```

T5 giả định:

```math
f(i,j) \approx f(j-i)
```

tức là attention phụ thuộc vào khoảng cách tương đối hơn là vị trí tuyệt đối.

---

# 5. Relative Position Bucketing

Thay vì học:

```math
2L-1
```

bias khác nhau.

T5 gom nhiều khoảng cách vào cùng một bucket.

```text
distance
    │
    ▼
┌────────────┐
│     0      │
├────────────┤
│    ±1      │
├────────────┤
│    ±2      │
├────────────┤
│    ±3      │
├────────────┤
│    ±4      │
├────────────┤
│   ±5~6     │
├────────────┤
│   ±7~10    │
├────────────┤
│  ±11~15    │
├────────────┤
│  ±16~31    │
├────────────┤
│  ±32~63    │
└────────────┘
```

---

# 6. Attention Equation

## Standard Attention

```math
A = \frac{QK^T}{\sqrt{d_k}}
```

## T5 Relative Bias Attention

```math
A
=
\frac{QK^T}{\sqrt{d_k}}
+
B
```

với:

```math
B_{ij} = E_{bucket(j-i)}
```

và:

```math
Y = Softmax(A)V
```

---

# 7. Multi-Head Relative Bias

```text
Head 1 ──► Bias Table 1
Head 2 ──► Bias Table 2
Head 3 ──► Bias Table 3
...
Head H ──► Bias Table H
```

Mỗi head có thể học:

* Local dependency
* Long-range dependency
* Syntax dependency
* Semantic dependency

---

# 8. Shared Across All Layers

```text
Transformer Layer 1
          │
          ├──────────────┐
Transformer Layer 2      │
          │              │
          ├──────────────┤
Transformer Layer 3      │
          │              │
          ├──────────────┤
          ⋮              │
Transformer Layer N      │
                         │
                         ▼
              Shared Relative Bias Table
```

Điều này giúp:

* giảm tham số;
* tăng tính nhất quán;
* cải thiện khả năng tổng quát hóa.

---

# 9. Độ phức tạp

```text
Standard Attention
O(n²d)

T5 Relative Bias
O(n²) lookup + addition
```

Không phát sinh:

* matrix multiplication;
* projection;
* tensor lớn bổ sung.

---

# 10. So sánh các phương pháp Position Encoding

| Method           | Relative | Extra Projection | Extrapolation |
| ---------------- | -------- | ---------------- | ------------- |
| Absolute PE      | ❌        | ❌                | Kém           |
| Shaw Relative    | ✅        | ✅                | Tốt           |
| T5 Relative Bias | ✅        | ❌                | Tốt           |
| ALiBi            | ✅        | ❌                | Rất tốt       |
| RoPE             | ✅        | ❌                | Rất tốt       |

---

# 11. Tổng kết một trang

```text
Input
  │
  ▼
Q,K,V Projection
  │
  ▼
QKᵀ / √dk
  │
  ├──────────────► Relative Distance (j-i)
  │                           │
  │                           ▼
  │                    Bucket Mapping
  │                           │
  │                           ▼
  │                    Learnable Bias B
  │                           │
  └───────────────(+ )────────┘
              │
              ▼
           Softmax
              │
              ▼
         Attention Weights
              │
              ▼
               V
              │
              ▼
            Output
```

---

# Công thức cuối cùng

```math
A_{ij} = \frac{Q_iK_j^\top} {\sqrt{d_k}} + E_{bucket(j-i)}
```

```math
Y= \text{softmax}(A)V
```

---

# Kết luận

```text
Absolute Position
        │
        ▼
Shaw Relative Attention
        │
        ▼
T5 Relative Position Bias
        │
        ├── ALiBi
        ├── RoPE
        ├── XPos
        └── Dynamic Position Bias
```

T5 Relative Position Bias là bước chuyển quan trọng từ **absolute positional encoding** sang **relative positional modeling**, đồng thời đặt nền móng cho phần lớn các kiến trúc positional encoding hiện đại trong các biến thể Transformer và `x-transformers`.
