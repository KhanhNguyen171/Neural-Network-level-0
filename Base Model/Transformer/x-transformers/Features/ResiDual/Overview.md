# ResiDual - Overview

> **Ý tưởng cốt lõi:** Kết hợp đồng thời **Pre-LayerNorm** và **Post-LayerNorm** để đạt được:

* Gradient ổn định của Pre-LN.
* Khả năng chống Representation Collapse của Post-LN.
* Huấn luyện Transformer rất sâu.
* Ổn định số học trong FP16/BF16.

---

# 1. Bức tranh tổng thể

```mermaid
flowchart TB

A["Input h_l"]

subgraph PRE["Pre-Norm Branch"]
B["LayerNorm"]
C["Sublayer F"]
D["Pre Residual Add"]
end

subgraph POST["Post-Norm Branch"]
E["LayerNorm"]
F["Post Residual Add"]
end

G["Output h_l+1"]

A --> B
B --> C
C --> D
A --> D
D --> E
E --> F
A --> F
F --> G
```

---

# 2. Công thức tổng quát

```math
\boxed{ h_{l+1} = h_l + LN \left( h_l + F \left( LN(h_l) \right) \right)}
```

---

# 3. Luồng thông tin của ResiDual

```mermaid
flowchart LR

X["Input h_l"]

X --> A["LN"]

A --> B["Attention / FFN"]

B --> C["+"]

X --> C

C --> D["LN"]

X --> E["Identity"]

D --> F["+"]

E --> F

F --> Y["Output h_l+1"]
```

---

# 4. Hai đường Residual

```mermaid
flowchart LR

A["Input"]

subgraph R1["Residual Path #1"]
B["Pre-LN"]
C["F(x)"]
D["x + F(x)"]
end

subgraph R2["Residual Path #2"]
E["LN(x + F(x))"]
F["x + LN(...)"]
end

A --> B
B --> C
C --> D
A --> D
D --> E
A --> F
E --> F
```

---

# 5. So sánh với Pre-LN và Post-LN

```mermaid
flowchart LR

subgraph POSTLN
A1["x"]
B1["F(x)"]
C1["x + F(x)"]
D1["LN"]
A1 --> B1
B1 --> C1
A1 --> C1
C1 --> D1
end

subgraph PRELN
A2["x"]
B2["LN"]
C2["F"]
D2["x + F(LN(x))"]
A2 --> B2
B2 --> C2
C2 --> D2
A2 --> D2
end

subgraph RESIDUAL
A3["x"]
B3["LN"]
C3["F"]
D3["x + F"]
E3["LN"]
F3["x + LN(...)"]

A3 --> B3
B3 --> C3
C3 --> D3
A3 --> D3
D3 --> E3
A3 --> F3
E3 --> F3
end
```

---

# 6. Phân tích Gradient

## Post-LN

```math
\frac{\partial h_{l+1}} {\partial h_l} = \frac{\partial LN}{\partial h} \left( I+\frac{\partial F}{\partial h_l} \right)
```

Gradient:

```math
\rightarrow 0
```

---

## Pre-LN

```math
\frac{\partial h_{l+1}} {\partial h_l} = I+ \frac{\partial F} {\partial h_l}
```

Gradient:

```math
\not\rightarrow 0
```

---

## ResiDual

```math
\frac{\partial h_{l+1}} {\partial h_l} = I + \frac{\partial LN} {\partial p_l} \left( I+ \frac{\partial F} {\partial h_l} \right)
```

Gradient:

```math
\not\rightarrow 0
```

và vẫn duy trì tính đa dạng của biểu diễn.

---

# 7. Representation Collapse

```mermaid
flowchart TB

A["Layer 1 Features"]
B["Layer 10 Features"]
C["Layer 50 Features"]
D["Layer 100 Features"]

A --> B --> C --> D

D --> E["Representation Collapse"]
```

---

# 8. ResiDual chống Collapse

```mermaid
flowchart TB

A["Layer 1"]

B["Layer 10"]

C["Layer 50"]

D["Layer 100"]

A --> B
B --> C
C --> D

B --> E["LayerNorm"]
C --> F["LayerNorm"]
D --> G["LayerNorm"]

E --> H["Diverse Features"]
F --> H
G --> H
```

---

# 9. Residual Scaling

Paper đề xuất:

```math
p_l = \alpha \left( h_l + F(LN(h_l)) \right)
```

với:

```math
\alpha = 0.1
```

nhằm:

* tránh overflow trong FP16;
* giữ nguyên hiệu quả mô hình vì:

```math
LN(cx)=LN(x)
```

---

# 10. Thuật toán Forward

```mermaid
flowchart TB

A["Input h_l"]

B["z = LN(h_l)"]

C["u = F(z)"]

D["p = h_l + u"]

E["q = LN(p)"]

F["h_l+1 = h_l + q"]

A --> B
B --> C
C --> D
A --> D
D --> E
E --> F
A --> F
```

---

# 11. Tổng kết

```mermaid
mindmap
  root((ResiDual))
    Motivation
      Vanishing Gradient
      Representation Collapse
    Architecture
      Pre LayerNorm
      Post LayerNorm
      Dual Residual
    Advantages
      Stable Optimization
      Deep Transformers
      Better Representations
      FP16 Stability
    Formula
      Dual Residual Formula
    x-Transformers
      resi_dual=True
      resi_dual_scale=0.1
```

```mermaid
flowchart TB

R["ResiDual"]

R --> M["Motivation"]
R --> A["Architecture"]
R --> P["Properties"]
R --> X["x-Transformers"]

M --> M1["Vanishing Gradient"]
M --> M2["Representation Collapse"]

A --> A1["Pre-LayerNorm"]
A --> A2["Post-LayerNorm"]
A --> A3["Dual Residual"]

P --> P1["Stable Gradients"]
P --> P2["Deep Transformers"]
P --> P3["FP16/BF16 Stability"]
P --> P4["Diverse Representations"]

X --> X1["resi_dual=True"]
X --> X2["resi_dual_scale=0.1"]
```

$$
h_{l+1} ​ =h_l ​+ LN(h_l​+F(LN(h_l​)))​
$$
---

# Key Takeaways

```text
Post-LN
 └── tốt về biểu diễn
 └── gradient không ổn định

Pre-LN
 └── gradient ổn định
 └── representation collapse

ResiDual
 └── kết hợp ưu điểm của cả hai
 └── phù hợp cho Transformer rất sâu
 └── được tích hợp trong x-transformers
```
