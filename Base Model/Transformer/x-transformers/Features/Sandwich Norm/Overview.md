# Sandwich Norm - Overview Diagram

> Minh họa tổng quát về động cơ, kiến trúc, công thức toán học, luồng tín hiệu và vai trò của **Sandwich Norm** trong các Transformer hiện đại.

---

# 1. Big Picture

```mermaid
flowchart TB

A[Transformer sâu]
A --> B[Residual Stream Drift]
A --> C[Activation Explosion]
A --> D[Gradient Instability]

B --> E[Pre-LN chưa kiểm soát đầu ra của Sublayer]
C --> E
D --> E

E --> F[Sandwich Norm]

F --> G[Chuẩn hóa đầu vào]
F --> H[Chuẩn hóa đầu ra]

G --> I[Residual ổn định]
H --> I

I --> J[Huấn luyện Transformer rất sâu]
```

---

# 2. Sự tiến hóa của Normalization trong Transformer

```mermaid
flowchart LR

A[Post-LayerNorm]
--> B[Pre-LayerNorm]
--> C[Sandwich Norm]
--> D[Residual Scaling]
--> E[DeepNorm]
--> F[NormFormer]
--> G[Transformer cực sâu]
```

---

# 3. So sánh kiến trúc

## Post-LayerNorm

```text
x
│
├── Sublayer
│
└── Add
      │
      LN
      │
      y
```

---

## Pre-LayerNorm

```text
x
│
├── LN
│
├── Sublayer
│
└── Add
      │
      y
```

---

## Sandwich Norm

```text
x
│
├── LN
│
├── Sublayer
│
├── LN
│
└── Add
      │
      y
```

---

# 4. Ý tưởng cốt lõi

```mermaid
flowchart LR

A[x]
--> B[LayerNorm]

B --> C[Sublayer]

C --> D[LayerNorm]

D --> E[Residual Add]

E --> F[y]
```

Sandwich Norm đặt:

```text
LayerNorm
      ↓
   Sublayer
      ↓
LayerNorm
```

tạo thành một "chiếc bánh sandwich" bao quanh sublayer.

---

# 5. Công thức toán học

## Pre-LayerNorm

```math
y=x+F(\mathrm{LN}(x))
```

---

## Sandwich Norm

```math
y=
x+
\mathrm{LN}
\left(
F(\mathrm{LN}(x))
\right)
```

---

# 6. Kiến trúc của một Transformer Block

```mermaid
flowchart TB

X[x_l]

X --> LN1[LayerNorm]

LN1 --> ATTN[Multi-Head Attention]

ATTN --> LN2[LayerNorm]

LN2 --> ADD1[Residual Add]

ADD1 --> LN3[LayerNorm]

LN3 --> FFN[Feed Forward]

FFN --> LN4[LayerNorm]

LN4 --> ADD2[Residual Add]

ADD2 --> OUT[x_l+1]
```

---

# 7. Phương trình đầy đủ

## Attention

```math
z_l= x_l+ \mathrm{LN} \left( \mathrm{Attention} ( \mathrm{LN}(x_l)) \right)
```

---

## Feed Forward

```math
x_{l+1} = z_l+ \mathrm{LN} \left( \mathrm{FFN} ( \mathrm{LN}(z_l) ) \right)
```

---

# 8. Luồng lan truyền tín hiệu

## Pre-LN

```mermaid
flowchart LR

A[x_l]
--> B[F_l]

B --> C[x_l + F_l]

C --> D[Norm tăng dần]
```

---

## Sandwich Norm

```mermaid
flowchart LR

A[x_l]
--> B[F_l]

B --> C[LayerNorm]

C --> D[x_l + F_l_hat]

D --> E[Norm ổn định]
```

---

# 9. Phân tích độ lớn của Residual Stream

## Không dùng Sandwich Norm

```text
Layer 1 : ||x|| = 10
Layer 2 : ||x|| = 16
Layer 3 : ||x|| = 24
Layer 4 : ||x|| = 38
Layer 5 : ||x|| = 61
```

---

## Có Sandwich Norm

```text
Layer 1 : ||x|| = 10
Layer 2 : ||x|| = 11
Layer 3 : ||x|| = 11.6
Layer 4 : ||x|| = 12
Layer 5 : ||x|| = 12.4
```

---

# 10. Cơ chế hoạt động

```mermaid
flowchart TB

A[Sublayer Output]

A --> B[LayerNorm]

B --> C[Variance ≈ 1]

C --> D[Magnitude ổn định]

D --> E[Residual Stream ổn định]

E --> F[Gradient ổn định]

F --> G[Huấn luyện sâu hơn]
```

---

# 11. Góc nhìn Gradient

```math
\frac{\partial L} {\partial x_l} = \frac{\partial L} {\partial x_{l+1}} \left( I+ \frac{\partial \hat F_l} {\partial x_l} \right)
```

với

```math
\hat F_l= \mathrm{LN} \left( F(\mathrm{LN}(x_l)) \right)
```

LayerNorm thứ hai giúp:

```math
\left\| \frac{\partial \hat F_l} {\partial x_l} \right\|
```

được kiểm soát tốt hơn.

---

# 12. Lợi ích đạt được

```mermaid
mindmap
root((Sandwich Norm))

  Ổn định Activation
    Giảm Activation Explosion
    Giảm Distribution Drift

  Ổn định Gradient
    Giảm Gradient Explosion
    Tối ưu dễ hơn

  Residual Stream
    Magnitude ổn định
    Variance ổn định

  Huấn luyện mô hình lớn
    Transformer sâu
    Large Scale Training
    x-transformers
```

---

# 13. Vị trí của Sandwich Norm trong x-transformers

```mermaid
flowchart LR

A[Pre-LN]
--> B[Sandwich Norm]
--> C[Residual Scaling]
--> D[DeepNorm]
--> E[NormFormer]
--> F[1000+ Layers Transformer]
```

---

# 14. Kết luận trực quan

```mermaid
flowchart TB

A[Input]
--> B[LayerNorm]

B --> C[Attention hoặc FFN]

C --> D[LayerNorm]

D --> E[Residual Add]

E --> F[Residual Stream ổn định]

F --> G[Gradient ổn định]

G --> H[Huấn luyện Transformer rất sâu]
```

---

# Tóm tắt một dòng

```text
Pre-LN:
x + F(LN(x))

Sandwich Norm:
x + LN(F(LN(x)))
```

> Một LayerNorm bổ sung sau mỗi residual branch giúp kiểm soát độ lớn activation, ổn định gradient và cải thiện khả năng mở rộng của Transformer sâu.
