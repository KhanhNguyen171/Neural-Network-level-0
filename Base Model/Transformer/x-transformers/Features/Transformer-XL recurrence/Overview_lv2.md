# Transformer-XL Recurrence Overview

> Tổng quan kiến trúc **Transformer-XL Recurrence Mechanism** bằng Mermaid để đưa trực tiếp vào `README.md` trên GitHub.


```mermaid
flowchart LR

subgraph Past
    P["Previous Segment"]
    M["Memory"]
end

subgraph Present
    X["Current Segment"]
    R["Relative Position Encoding"]
    A["Multi Head Attention"]
    F["Feed Forward"]
    H["Hidden States"]
end

P --> M
M --> A
X --> A
R --> A
A --> F
F --> H
H --> M

style M fill:#ffe699
style A fill:#b4c7e7
style H fill:#c6e0b4
```

---

# 1. Từ Transformer chuẩn đến Transformer-XL

```mermaid
flowchart LR
    A[Token x1 ... xL] --> B[Standard Transformer]
    B --> C[Context Window = L]

    D[Long Sequence x1 ... x10000] --> E[Transformer-XL]
    E --> F[Memory M]
    E --> G[Current Segment L]
    F --> H[Effective Context = L + M]
    G --> H
```

---

# 2. Segment-Level Recurrence

```mermaid
flowchart LR
    S1["Segment 1<br>x1 ... x512"]
    M1["Memory 1"]
    S2["Segment 2<br>x513 ... x1024"]
    M2["Memory 2"]
    S3["Segment 3<br>x1025 ... x1536"]

    S1 --> M1
    M1 --> S2
    S2 --> M2
    M2 --> S3
```

---

# 3. Memory Recurrence Mechanism

```mermaid
flowchart TD
    H["Hidden States from Previous Segment"]
    D["Detach Operation"]
    M["Memory Bank"]
    S["Current Segment"]

    H --> D
    D --> M
    M --> S
```

---

# 4. Attention Architecture

```mermaid
flowchart TD
    Q["Query from Current Segment"]
    M["Memory States"]
    C["Current States"]

    KV["Keys and Values"]
    A["Scaled Dot Product Attention"]
    O["Attention Output"]

    M --> KV
    C --> KV

    Q --> A
    KV --> A

    A --> O
```

---

# 5. Effective Context Expansion

```mermaid
flowchart LR
    M["Memory M Tokens"]
    L["Current Segment L Tokens"]

    E["Effective Context"]

    M --> E
    L --> E
```

---

# 6. Information Flow Across Segments

```mermaid
flowchart LR
    S1[Segment 1]
    M1[Memory 1]
    S2[Segment 2]
    M2[Memory 2]
    S3[Segment 3]

    S1 --> M1
    M1 --> S2
    S2 --> M2
    M2 --> S3
```

---

# 7. Forward vs Backward Propagation

```mermaid
flowchart LR
    P1[Past Segment]
    P2[Current Segment]
    P3[Future Segment]

    P1 --> P2
    P2 --> P3

    B1[Gradient]
    B2[Gradient]

    P3 -.-> B2
    B2 -.-> P2

    style P1 fill:#d9d9d9
```

**Lưu ý**

Memory được:

```python
memory = memory.detach()
```

nên gradient không đi xuyên qua các segment cũ.

---

# 8. Relative Positional Encoding

```mermaid
flowchart LR
    Xi[Token xi]
    Xj[Token xj]

    Xi <-->|distance i-j| Xj
```

Attention score:

$$
A_{i,j}= q_i^Tk_j + q_i^Tr_{i-j} + u^Tk_j + v^Tr_{i-j}
$$

---

# 9. Training Pipeline

```mermaid
flowchart TD
    A[Long Sequence]
    B[Segment 1]
    C[Memory 1]
    D[Segment 2]
    E[Memory 2]
    F[Segment 3]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

---

# 10. Inference Pipeline

```mermaid
flowchart LR
    P[Prompt]
    S1[Segment 1]
    M1[Memory 1]
    S2[Segment 2]
    M2[Memory 2]
    G[Generate Next Tokens]

    P --> S1
    S1 --> M1
    M1 --> S2
    S2 --> M2
    M2 --> G
```

---

# 11. x-transformers Implementation

```mermaid
flowchart TD
    A[Input Segment]
    B[TransformerWrapper]
    C[Decoder]
    D[Relative Position Bias]
    E[Hidden States]
    F[return_mems=True]
    G[Memory]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

---

# 12. Internal Transformer-XL Block

```mermaid
flowchart TD
    MEM["Memory"]
    X["Current Tokens"]

    ATTN["Multi Head Attention"]
    ADD1["Residual and LayerNorm"]
    FFN["Feed Forward Network"]
    ADD2["Residual and LayerNorm"]
    H["Hidden States"]
    DET["Detach"]
    NEWM["New Memory"]

    MEM --> ATTN
    X --> ATTN

    ATTN --> ADD1
    ADD1 --> FFN
    FFN --> ADD2
    ADD2 --> H
    H --> DET
    DET --> NEWM
```

---

# 13. Complete Architecture Overview

```mermaid
flowchart LR

subgraph Previous_Segment
    A["Hidden States"]
end

subgraph Memory
    B["Memory Bank"]
end

subgraph Current_Segment
    C["Input Tokens"]
    D["Relative Position Encoding"]
    E["Multi Head Attention"]
    F["Feed Forward"]
    G["Hidden States"]
end

A --> B
B --> E
C --> E
D --> E
E --> F
F --> G
G --> B
```

---

# 14. Big Picture

```mermaid
flowchart TD

TXL["Transformer XL"]

REC["Segment Level Recurrence"]
MEM["Memory Reuse"]
POS["Relative Position Encoding"]
LONG["Long Context Modeling"]
TRAIN["Detach and Truncated BPTT"]
COMP["Complexity O of L times L plus M"]
XTR["x transformers"]

TXL --> REC
TXL --> MEM
TXL --> POS
TXL --> LONG
TXL --> TRAIN
TXL --> COMP
TXL --> XTR
```

---

# Key Takeaways

```text
Transformer
┌────────────────────┐
│ Context = L        │
└────────────────────┘

Transformer-XL
┌────────────────────┐
│ Memory = M         │
├────────────────────┤
│ Current = L        │
└────────────────────┘

Effective Context = L + M

✓ Segment-Level Recurrence
✓ Memory Reuse
✓ Relative Position Encoding
✓ Long-Range Dependency Modeling
✓ Foundation of Modern Long-Context Transformers
```
