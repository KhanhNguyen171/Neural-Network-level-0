# Cosine Similarity Attention - Overview

> Tổng quan kiến trúc, nguyên lý toán học, cơ chế ổn định số học và vị trí của **Cosine Similarity Attention (QK-Norm Attention)** trong họ kiến trúc x-Transformers.

---

# 1. Big Picture

```mermaid
flowchart LR

X[Input Embeddings]

X --> Q[Linear Projection WQ]
X --> K[Linear Projection WK]
X --> V[Linear Projection WV]

Q --> QN[L2 Normalize Queries]
K --> KN[L2 Normalize Keys]

QN --> COS[Cosine Similarity]
KN --> COS

COS --> TEMP[Learnable Temperature s]

TEMP --> SM[Softmax]

SM --> ATTN[Attention Weights]

V --> OUT

ATTN --> OUT[Weighted Sum]

OUT --> Y[Output]
```

---

# 2. Core Mathematical Idea

Transformer chuẩn:

```math
\mathrm{Attention}(Q,K,V) = \mathrm{Softmax} \left( \frac{QK^T}{\sqrt d} \right)V
```

Cosine Similarity Attention:

```math
\mathrm{Attention}(Q,K,V) = \mathrm{Softmax} \left( s\hat Q\hat K^T \right)V
```

với

```math
\hat q = \frac{q}{\|q\|_2}, \qquad \hat k = \frac{k}{\|k\|_2}
```

---

# 3. Geometric Interpretation

Attention score trở thành:

```math
\hat q^T\hat k = \cos\theta
```

Do đó:

```math
-1 \le \cos\theta \le 1
```

attention score luôn bị chặn.

---

# 4. Dot Product vs Cosine Similarity

```mermaid
flowchart TB

subgraph Dot_Product_Attention
A1[q]
A2[k]

A1 --> B1[qᵀk]
A2 --> B1

B1 --> C1["||q|| ||k|| cos(θ)"]

C1 --> D1[Unbounded Logits]
D1 --> E1[Softmax Saturation]
E1 --> F1[Training Instability]
end

subgraph Cosine_Attention
G1[q]
G2[k]

G1 --> H1[L2 Norm]
G2 --> I1[L2 Norm]

H1 --> J1[cos θ]
I1 --> J1

J1 --> K1[Bounded Logits]
K1 --> L1[Stable Softmax]
L1 --> M1[Stable Training]
end
```

---

# 5. Information Flow

```mermaid
sequenceDiagram

participant X as Input
participant Q as Queries
participant K as Keys
participant N as L2 Norm
participant S as Similarity
participant A as Softmax
participant V as Values

X->>Q: WQ
X->>K: WK
Q->>N: Normalize
K->>N: Normalize
N->>S: Cosine Similarity
S->>A: Scale by s
A->>V: Attention Weights
V-->>X: Weighted Sum
```

---

# 6. Why Does It Work?

```mermaid
flowchart TD

A[Large Transformer]

A --> B[Large Query-Key Norms]

B --> C[Large Attention Logits]

C --> D[Softmax Saturation]

D --> E[Vanishing Gradients]

E --> F[Training Instability]

A --> G[L2 Normalization]

G --> H[Bounded Similarity]

H --> I[Stable Softmax]

I --> J[Stable Optimization]
```

---

# 7. Numerical Stability

## Standard Attention

```math
q^Tk = \|q\| \|k\| \cos\theta
```

Range:

```math
(-\infty,+\infty)
```

---

## Cosine Attention

```math
\hat q^T\hat k = \cos\theta
```

Range:

```math
[-1,1]
```

---

## Grouped QK-Norm

```math
-G \le \hat q^T\hat k \le G
```

---

# 8. Role of Temperature

```mermaid
flowchart LR

A[Cosine Similarity]

A --> B1[s small]
A --> B2[s large]

B1 --> C1[Soft Attention]

B2 --> C2[Sharp Attention]
```

---

# 9. Grouped QK Normalization

```mermaid
flowchart LR

Q[Head Dimension]

Q --> G1[Group 1]
Q --> G2[Group 2]
Q --> G3[Group 3]
Q --> G4[Group G]

G1 --> N1[L2 Norm]
G2 --> N2[L2 Norm]
G3 --> N3[L2 Norm]
G4 --> N4[L2 Norm]

N1 --> S
N2 --> S
N3 --> S
N4 --> S

S[Similarity Range = -G ... G]
```

---

# 10. Spectral View

```mermaid
flowchart TB

A[Normalize Q and K]

A --> B[Bound Operator Norm]

B --> C[Stable Jacobian]

C --> D[Controlled Gradients]

D --> E[Deep Transformer Training]
```

---

# 11. Computational Complexity

| Method                      | Time   | Memory |
| --------------------------- | ------ | ------ |
| Dot Product Attention       | O(n²d) | O(n²)  |
| Cosine Similarity Attention | O(n²d) | O(n²)  |

Additional cost:

```math
O(nd)
```

for L2 normalization.

---

# 12. Position in x-Transformers

```mermaid
flowchart LR

A[Original Transformer]
--> B[Pre-LN Transformer]
--> C[NormFormer]
--> D[Cosine Similarity Attention]
--> E[QK-Norm]
--> F[Grouped QK-Norm]
--> G[Stable Deep Transformers]
```

---

# 13. Complete Architecture

```mermaid
flowchart TB

subgraph Input
X[Input Tokens]
end

subgraph Projections
Q[Queries]
K[Keys]
V[Values]
end

subgraph Cosine Attention
QN[L2 Normalize Q]
KN[L2 Normalize K]
SIM[Cosine Similarity]
TEMP[Temperature s]
SM[Softmax]
end

subgraph Output
W[Weighted Sum]
Y[Output]
end

X --> Q
X --> K
X --> V

Q --> QN
K --> KN

QN --> SIM
KN --> SIM

SIM --> TEMP
TEMP --> SM

SM --> W
V --> W

W --> Y
```

---

# 14. Summary

```mermaid
mindmap
root((Cosine Similarity Attention))

    Motivation
        Stable Training
        Avoid Overflow
        Prevent Softmax Saturation

    Mathematics
        L2 Normalize Q
        L2 Normalize K
        Cosine Similarity
        Learnable Temperature

    Advantages
        Bounded Logits
        Better Gradients
        FP16 Friendly
        Deep Transformers

    Extensions
        QK-Norm
        Grouped QK-Norm
        Stable ViT
        x-Transformers
```

---

# Key Equation

```math
\boxed{ \mathrm{Attention}(Q,K,V) = \mathrm{Softmax} \left( s\hat Q\hat K^T \right)V }
```

Cosine Similarity Attention biến attention từ phép đo phụ thuộc vào **độ lớn vector** thành phép đo phụ thuộc vào **góc giữa các vector**, từ đó cải thiện đáng kể độ ổn định số học và khả năng mở rộng của Transformer quy mô lớn.
