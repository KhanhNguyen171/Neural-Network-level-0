# Cross Attention – Overview

> Minh họa tổng quát kiến trúc **Cross Attention** trong `x-transformers`, nguyên lý toán học, luồng dữ liệu và vai trò trong các kiến trúc Transformer hiện đại.

---

# 1. Cross Attention là gì?

Cross Attention là cơ chế Attention mà:

* **Query (Q)** được sinh từ một chuỗi đích (**Target Sequence**).
* **Key (K)** và **Value (V)** được sinh từ một chuỗi ngữ cảnh (**Context Sequence**).

Khác với Self-Attention:

| Cơ chế          | Query | Key | Value |
| --------------- | ----- | --- | ----- |
| Self-Attention  | X     | X   | X     |
| Cross Attention | X     | C   | C     |

trong đó:

* $X$: Target Sequence
* $C$: Context Sequence

---

# 2. Ý tưởng cốt lõi

```text
Target Sequence
      │
      └───► Query

Context Sequence
      ├───► Key
      └───► Value
```

Mục tiêu:

```math
Target Tokens
      ↓
Tìm kiếm thông tin
      ↓
Context Tokens
      ↓
Tổng hợp biểu diễn mới
```

---

# 3. Kiến trúc tổng quát

```mermaid
flowchart LR

X[Target Sequence]
C[Context Sequence]

X --> Q[Query Projection]

C --> K[Key Projection]
C --> V[Value Projection]

Q --> A[Attention Scores]
K --> A

A --> S[Softmax]

S --> O[Weighted Sum]
V --> O

O --> Y[Cross Attention Output]
```

---

# 4. Công thức toán học

## Bước 1: Sinh Query

```math
Q = XW_Q
```

## Bước 2: Sinh Key và Value

```math
K = CW_K
```

```math
V = CW_V
```

## Bước 3: Tính Attention Score

```math
S = \frac{QK^T}{\sqrt{d_k}}
```

## Bước 4: Chuẩn hóa

```math
A = softmax(S)
```

## Bước 5: Tổng hợp thông tin

```math
O = AV
```

---

# 5. Pipeline tính toán

```mermaid
flowchart TD

A[Target Sequence]
B[Context Sequence]

A --> C[Linear Q]

B --> D[Linear K]
B --> E[Linear V]

C --> F[QKᵀ]
D --> F

F --> G[Scale by sqrt(d)]
G --> H[Mask]

H --> I[Softmax]

I --> J[Attention Weights]

E --> K[Weighted Sum]

J --> K

K --> L[Output]
```

---

# 6. Multi-Head Cross Attention

```mermaid
flowchart LR

Q[Query]
K[Key]
V[Value]

Q --> H1[Head 1]
K --> H1
V --> H1

Q --> H2[Head 2]
K --> H2
V --> H2

Q --> H3[Head h]
K --> H3
V --> H3

H1 --> C[Concat]
H2 --> C
H3 --> C

C --> O[Linear Projection]
```

---

# 7. Kích thước Tensor

Cho:

```text
B : Batch size
N : Target length
M : Context length
D : Hidden dimension
H : Number of heads
d = D/H
```

### Query

```math
Q \in \mathbb{R}^{B\times N\times D}
```

### Key

```math
K \in \mathbb{R}^{B\times M\times D}
```

### Value

```math
V \in \mathbb{R}^{B\times M\times D}
```

### Attention Matrix

```math
A \in \mathbb{R}^{B\times H\times N\times M}
```

### Output

```math
O \in \mathbb{R}^{B\times N\times D}
```

---

# 8. Cross Attention vs Self Attention

```mermaid
flowchart TB

subgraph SelfAttention
A1[X]
A1 --> A2[Q]
A1 --> A3[K]
A1 --> A4[V]
end

subgraph CrossAttention
B1[Target]
B2[Context]

B1 --> B3[Q]
B2 --> B4[K]
B2 --> B5[V]
end
```

| Thuộc tính                   | Self Attention | Cross Attention |
| ---------------------------- | -------------- | --------------- |
| Nguồn Query                  | Chính nó       | Target          |
| Nguồn Key                    | Chính nó       | Context         |
| Nguồn Value                  | Chính nó       | Context         |
| Tích hợp nhiều nguồn dữ liệu | Không          | Có              |
| Encoder-Decoder              | Không          | Có              |

---

# 9. Độ phức tạp tính toán

Attention Matrix:

```math
A \in \mathbb{R}^{N\times M}
```

Chi phí:

### FLOPs

```math
O(NMd)
```

### Bộ nhớ

```math
O(NM)
```

Nếu:

```math
M \gg N
```

thì chi phí chủ yếu nằm ở context sequence.

---

# 10. Cross Attention trong Encoder-Decoder

```mermaid
flowchart LR

S[Source Sequence]
E[Encoder]
C[Encoded Context]

T[Decoder Tokens]

S --> E
E --> C

T --> CA[Cross Attention]
C --> CA

CA --> O[Decoder Output]
```

---

# 11. Cross Attention trong Retrieval Transformer

```mermaid
flowchart LR

Q[Query]
R[Retrieved Memory]

R --> E[Memory Encoder]
E --> M[Context Memory]

Q --> CA[Cross Attention]
M --> CA

CA --> O[Prediction]
```

---

# 12. Cross Attention trong Graph Transformer

```mermaid
flowchart LR

N[Node]

NB[Neighbor Nodes]

NB --> ENC[Neighbor Encoder]

ENC --> CA[Cross Attention]

N --> CA

CA --> OUT[Updated Node]
```

Phương trình Message Passing:

```math
h_i' = \sum_j \alpha_{ij} W_V h_j
```

với:

```math
\alpha_{ij} = softmax \left( \frac{q_i^Tk_j} {\sqrt d} \right)
```

---

# 13. Cross Attention trong Multi-Modal Learning

```mermaid
flowchart LR

IMG[Image Features]
TXT[Text Features]

IMG --> CA[Cross Attention]
TXT --> CA

CA --> F[Fused Representation]
```

---

# 14. Cross Attention trong x-transformers

```python
model(
    x,
    context=context,
    mask=mask,
    context_mask=context_mask
)
```

Ý nghĩa:

```text
x               → Query Source
context         → Key/Value Source
mask            → Target Mask
context_mask    → Context Mask
```

---

# 15. Thuật toán tổng quát

```text
Input:
    X : Target Sequence
    C : Context Sequence

Q = XWQ
K = CWK
V = CWV

S = QKᵀ / sqrt(d)

S = S + Mask

A = Softmax(S)

O = AV

return O
```

---

# 16. Tổng kết kiến trúc

```mermaid
flowchart TD

A[Target Sequence]
B[Context Sequence]

A --> Q[Query]

B --> K[Key]
B --> V[Value]

Q --> ATTN[Cross Attention]
K --> ATTN
V --> ATTN

ATTN --> O[Output Representation]

O --> APP1[Encoder Decoder]
O --> APP2[Retrieval]
O --> APP3[Graph Transformer]
O --> APP4[Multi Modal]
O --> APP5[Memory Transformer]
```

---

# Ghi nhớ

* Query đến từ **Target**.
* Key và Value đến từ **Context**.
* Cho phép một tập token truy xuất thông tin từ tập token khác.
* Là thành phần nền tảng của:

  * Encoder-Decoder Transformer.
  * Retrieval-Augmented Transformer.
  * Graph Transformer.
  * Multi-modal Transformer.
  * Memory-Augmented Transformer.
  * Perceiver Architecture.

Cross Attention là cơ chế giúp Transformer tiến hóa từ mô hình xử lý một chuỗi đơn lẻ thành kiến trúc học biểu diễn tổng quát có khả năng tích hợp nhiều nguồn thông tin khác nhau.
