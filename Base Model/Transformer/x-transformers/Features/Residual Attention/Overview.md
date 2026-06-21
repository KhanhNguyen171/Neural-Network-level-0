# Residual Attention (RealFormer) Overview

## 1. Ý tưởng cốt lõi

Transformer chuẩn:

```math
S_l=\frac{Q_lK_l^T}{\sqrt{d}}
```

RealFormer:

```math
R_l=S_l+R_{l-1}
```

```math
P_l= Softmax (R_l)
```

Trong đó:

* $S_l$: attention logits hiện tại.
* $R_l$: residual attention logits.
* $P_l$: attention probability.

---

# 2. Tổng quan kiến trúc

```mermaid
flowchart TD

A[Input X] --> B[Linear Projection]
B --> C[Q]
B --> D[K]
B --> E[V]

C --> F[QKᵀ / √d]
D --> F

G[Residual Attention<br>R(l-1)] --> H[Add]

F --> H

H --> I[Residual Logits<br>R(l)]

I --> J[Softmax]
J --> K[Attention Matrix]
K --> L[Multiply V]
E --> L
L --> M[Attention Output]
```

---

# 3. So sánh với Transformer chuẩn

## Vanilla Transformer

```mermaid
flowchart LR

A[Q,K] --> B[QKᵀ / √d]
B --> C[Softmax]
C --> D[Attention]
D --> E[Output]
```

---

## Residual Attention (RealFormer)

```mermaid
flowchart LR

A[Q,K] --> B[QKᵀ / √d]
F[Previous Attention Logits] --> C[Add]
B --> C
C --> D[Softmax]
D --> E[Attention]
E --> G[Output]
```

---

# 4. Attention được tích lũy qua các tầng

```mermaid
flowchart TB

A[S1] --> B[R1 = S1]
B --> C[R2 = S2 + R1]
C --> D[R3 = S3 + R2]
D --> E[R4 = S4 + R3]
```

Hay:

```math
R_l=\sum_{i=1}^{l}S_i
```

---

# 5. Minh họa trực quan nhiều tầng

```text
Layer 1

QKᵀ/√d
    │
    ▼
   S1
    │
    ▼
 softmax
    │
    ▼
 Output1


Layer 2

QKᵀ/√d
    │
    ▼
   S2
    │
    ▼
 + R1
    │
    ▼
   R2
    │
    ▼
 softmax
    │
    ▼
 Output2


Layer 3

QKᵀ/√d
    │
    ▼
   S3
    │
    ▼
 + R2
    │
    ▼
   R3
    │
    ▼
 softmax
    │
    ▼
 Output3
```

---

# 6. Góc nhìn Residual Learning

```mermaid
flowchart TB

A[Layer l-1 Attention]
--> B[Residual Attention]

C[Current Layer Score]
--> B

B --> D[Learn Small Correction]
```

Tương tự ResNet:

```math
x_{l+1}=x_l+F(x_l)
```

RealFormer:

```math
R_l=R_{l-1}+S_l
```

Mỗi layer chỉ cần học:

```math
\Delta Attention
```

thay vì học lại toàn bộ attention map.

---

# 7. Gradient Flow

```mermaid
flowchart BT

A[Loss]
--> B[Layer L]

B --> C[Layer L-1]

C --> D[Layer L-2]

D --> E[Layer L-3]
```

Residual Attention tạo thêm đường truyền gradient:

```text
Loss
 ↓
R(L)
 ↓
R(L-1)
 ↓
R(L-2)
 ↓
R(L-3)
```

Giúp:

* giảm vanishing gradient;
* tăng tính ổn định khi huấn luyện Transformer sâu;
* tăng tốc hội tụ.

---

# 8. Tại sao RealFormer hoạt động?

```mermaid
mindmap
root((Residual Attention))
    Stable Attention
        Preserve dependencies
        Less oscillation
    Better Optimization
        Easier convergence
        Higher learning rate
    Better Gradient Flow
        Deep Transformer
        Stable training
    Sparse Attention
        Lower entropy
        Focus on important tokens
    Zero Parameters
        No extra weights
        Minimal modification
```

---

# 9. Tổng quan toàn bộ kiến trúc

```mermaid
flowchart TB

subgraph Layer1
A1[QKᵀ/√d]
A2[Softmax]
A3[Attention Output]
A1 --> A2
A2 --> A3
end

subgraph Layer2
B1[QKᵀ/√d]
B2[+ Residual Logits]
B3[Softmax]
B4[Attention Output]
B1 --> B2
B2 --> B3
B3 --> B4
end

subgraph Layer3
C1[QKᵀ/√d]
C2[+ Residual Logits]
C3[Softmax]
C4[Attention Output]
C1 --> C2
C2 --> C3
C3 --> C4
end

A1 -. Residual Attention .-> B2
B2 -. Residual Attention .-> C2
```

---

# 10. Độ phức tạp

| Thành phần    | Vanilla  | RealFormer |
| ------------- | -------- | ---------- |
| Parameters    | $P$      | $P$        |
| FLOPs         | $O(n^2)$ | $O(n^2)$   |
| Memory        | $O(n^2)$ | $O(n^2)$   |
| Extra Weights | 0        | 0          |

---

# 11. Tóm tắt

```text
Transformer
      │
      ▼
Attention Scores
      │
      ▼
Residualize Attention Logits
      │
      ▼
Stable Attention Distribution
      │
      ▼
Better Gradient Flow
      │
      ▼
Deep Transformer Training
      │
      ▼
Better Performance
```

---

# Key Equation

```math
S_l=\frac{Q_lK_l^T}{\sqrt{d}}
```

```math
R_l=S_l+R_{l-1}
```

```math
P_l= Softmax (R_l)
```

```math
O_l=P_lV_l
```

---

# References

```bibtex
@article{he2021realformer,
  title={RealFormer: Transformer Likes Residual Attention},
  author={He, Ruining and Ravula, Anirudh and Kanagal, Bhargav and Ainslie, Joshua},
  journal={arXiv preprint arXiv:2012.11747},
  year={2021}
}
```
