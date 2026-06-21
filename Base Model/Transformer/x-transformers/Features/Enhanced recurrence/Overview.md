# Enhanced Recurrence in x-Transformers: Overview

> **Core Idea**

```math
M_l^{(t)} = SG \left( H_{l+s}^{(t-1)} \right)
```

instead of

```math
M_l^{(t)} = SG \left( H_l^{(t-1)} \right)
```

where:

* $H_l^{(t-1)}$: hidden states của đoạn trước,
* $M_l^{(t)}$: bộ nhớ của đoạn hiện tại,
* $s$: số tầng dịch bộ nhớ xuống,
* $SG$: phép dừng gradient (stop-gradient).

---

# 1. Bức tranh tổng quan

```mermaid
flowchart LR

A[Input Segment t-1]
--> B[Transformer Layers]

B --> C[Hidden States]
C --> D[Stored Memories]

D --> E[Cross-Layer Memory Routing]

E --> F[Input Segment t]
F --> G[Transformer Layers]
G --> H[Predictions]
```

---

# 2. Sự tiến hóa của cơ chế hồi quy (Recurrence)

```mermaid
flowchart TB

subgraph Vanilla Transformer
A1[Chỉ dùng ngữ cảnh hiện tại]
end

subgraph TransformerXL
A2[Segment hiện tại]
A3[Bộ nhớ từ segment trước]
A3 --> A2
end

subgraph Enhanced Recurrence
A4[Segment hiện tại]
A5[Bộ nhớ ngữ nghĩa từ quá khứ]
A5 --> A4
A6[Định tuyến liên tầng]
A6 --> A4
end
```

---

# 3. Transformer-XL vs Enhanced Recurrence

## Transformer-XL

```mermaid
flowchart LR

subgraph Segment_t_minus_1
L11[Layer1]
L12[Layer2]
L13[Layer3]
L14[Layer4]
end

subgraph Segment_t
N11[Layer1]
N12[Layer2]
N13[Layer3]
N14[Layer4]
end

L11 --> N11
L12 --> N12
L13 --> N13
L14 --> N14
```

Bộ nhớ quay về __đúng tầng tương ứng__.

---

## Enhanced Recurrence

```mermaid
flowchart LR

subgraph Segment_t_minus_1
L21[Layer1]
L22[Layer2]
L23[Layer3]
L24[Layer4]
end

subgraph Segment_t
N21[Layer1]
N22[Layer2]
N23[Layer3]
N24[Layer4]
end

L24 --> N23
L23 --> N22
L22 --> N21
```

Bộ nhớ được __định tuyến xuống tầng thấp hơn.__

---

# 4. Cross-Layer Information Flow

```mermaid
flowchart TB

Semantic[Đặc trưng ngữ nghĩa mức cao]
--> Routing[Đẩy bộ nhớ xuống dưới]

Routing
--> Lower[Tầng thấp]

Lower
--> Attention[Cơ chế Self-Attention]

Attention
--> Prediction[Dự đoán token tiếp theo]
```

---

# 5. Complete Architecture

```mermaid
flowchart TB

subgraph Previous Segment
P1[Layer 1]
P2[Layer 2]
P3[Layer 3]
P4[Layer 4]
P5[Layer 5]
P6[Layer 6]
end

subgraph Current Segment
C1[Layer 1]
C2[Layer 2]
C3[Layer 3]
C4[Layer 4]
C5[Layer 5]
C6[Layer 6]
end

P6 --> C5
P5 --> C4
P4 --> C3
P3 --> C2
P2 --> C1

C1 --> C2
C2 --> C3
C3 --> C4
C4 --> C5
C5 --> C6
```

---

# 6. Information Propagation

## Transformer-XL

```mermaid
flowchart TB

A[Trạng thái ngữ nghĩa quá khứ]
--> B[Cùng tầng]

B --> C[Nhiều bước hồi quy]

C --> D[Dự đoán hiện tại]
```

---

## Enhanced Recurrence

```mermaid
flowchart TB

A[Trạng thái ngữ nghĩa quá khứ]
--> B[Tầng thấp hơn]

B --> C[Dự đoán hiện tại]
```

Shorter path:

```math
\text{Path}_{ER}
<
\text{Path}_{TXL}
```

---

# 7. Attention Computation

```mermaid
flowchart LR

M[Bộ nhớ]
--> K

H[Hidden states hiện tại]
--> Q
H --> K
H --> V

Q[Query]
K[Key]
V[Value]

Q --> ATTN[Scaled Dot-Product Attention]
K --> ATTN
V --> ATTN

ATTN --> OUT[Output]
```

where

```math
Q=W_QH
```

```math
K=W_K[M;H]
```

```math
V=W_V[M;H]
```

---

# 8. Algorithm Overview

```mermaid
flowchart TB

A[Tính hidden states]
--> B[Lưu bộ nhớ]

B --> C[Dịch bộ nhớ xuống]

C --> D[Đưa bộ nhớ sang segment tiếp theo]

D --> E[Tính segment mới]
```

---

# 9. x-Transformers Implementation

```mermaid
flowchart LR

A[Segment t-1]
--> B[return_mems=True]

B --> C[mems]

C --> D[shift_mem_down]

D --> E[Segment t]
```

Implementation:

```python
shift_mem_down = 1
```

gives

```text
Layer6 → Layer5
Layer5 → Layer4
Layer4 → Layer3
Layer3 → Layer2
Layer2 → Layer1
```

---

# 10. Computational Cost

```mermaid
mindmap
  root((Enhanced Recurrence))
    Tham số
      Không tăng
    FLOPs
      Không tăng
    Bộ nhớ
      Không tăng
    Độ dài ngữ cảnh
      Tăng
    Phụ thuộc dài hạn
      Tốt hơn
    Lan truyền ngữ nghĩa
      Nhanh hơn
```

---

# 11. Position in the Evolution of x-Transformers

```mermaid
flowchart LR

TXL[Transformer-XL]
--> ER[Enhanced Recurrence]

ER
--> RA[Residual Attention]

ER
--> HC[Hyper Connections]

ER
--> MEM[Memory-Augmented Transformers]

ER
--> LONG[Long Context LLMs]
```

---

# Final Summary

```mermaid
mindmap
  root((Enhanced Recurrence))
    Ý tưởng chính
      Định tuyến bộ nhớ liên tầng
    Công thức
      M_l = H_ l+s 
    Lợi ích
      Lan truyền ngữ nghĩa nhanh hơn
      Ngữ cảnh hiệu dụng dài hơn
      Cải thiện phụ thuộc dài hạn
      Không tăng tham số
      Không tăng FLOPs
    Triển khai
      shift_mem_down
    Ứng dụng
      x-transformers
      Long-context models
      Memory-augmented models
```
