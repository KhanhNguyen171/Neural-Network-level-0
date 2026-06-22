# Shifted Tokens (Time-Mixing) — Overview Diagram

---

# 1. Big Picture

```mermaid
flowchart LR

X["Input Tokens"]

X --> S["Shifted Tokens<br/>(Time-Mixing)"]

S --> A["Self-Attention"]
S --> F["Feed Forward Network"]

A --> O["Transformer Block Output"]
F --> O

O --> N["Next Transformer Layer"]
```

---

# 2. Core Idea

Thay vì sử dụng hoàn toàn token hiện tại:

$$
x_t
$$

Shifted Tokens xây dựng:

$$
\tilde{x}*t= [x_t^{(0)}, x*{t-1}^{(1)}, x_{t-2}^{(2)}, \dots]
$$

để đưa thông tin quá khứ trực tiếp vào biểu diễn hiện tại.

---

# 3. Token Shift Mechanism

```mermaid
flowchart TB

subgraph Sequence
T0["x(t-2)"]
T1["x(t-1)"]
T2["x(t)"]
end

subgraph Shift
C0["Chunk 0<br/>No Shift"]
C1["Chunk 1<br/>Shift by 1"]
C2["Chunk 2<br/>Shift by 2"]
end

T2 --> C0
T1 --> C1
T0 --> C2

C0 --> M["Mixed Representation"]
C1 --> M
C2 --> M
```

---

# 4. Feature Mixing

```text
Original

x(t)
┌────────────────────────┐
│ d0 d1 d2 d3 d4 d5 d6 d7 │
└────────────────────────┘


Shift = 1

x(t)
┌────────────────────────┐
│ current │ previous      │
└────────────────────────┘


Shift = 2

x(t)
┌────────────────────────┐
│ current │ t-1 │ t-2     │
└────────────────────────┘
```

---

# 5. Mathematical Formulation

Given:

$$
X \in \mathbb{R}^{L \times d}
$$

Split feature dimension into:

$$
X= [X^{(0)},X^{(1)},...,X^{(s)}]
$$

Apply temporal shifts:

$$
\tilde{X}^{(i)}= \text{Shift}(X^{(i)},i)
$$

Concatenate:

$$
\tilde{X}= [\tilde{X}^{(0)}, \tilde{X}^{(1)}, \dots, \tilde{X}^{(s)}]
$$

---

# 6. Data Flow inside Transformer

```mermaid
flowchart TD

I["Input Embedding"]

S["Shifted Tokens"]

LN["LayerNorm"]

ATTN["Multi-Head Attention"]

FFN["Feed Forward"]

RES["Residual Connection"]

O["Output"]

I --> S
S --> LN
LN --> ATTN
ATTN --> FFN
FFN --> RES
RES --> O
```

---

# 7. Gradient Path Comparison

## Standard Transformer

```mermaid
flowchart LR

A["x(t-1)"]
B["Attention"]
C["x(t)"]

A --> B --> C
```

---

## Shifted Tokens

```mermaid
flowchart LR

A["x(t-1)"]

B["Concatenation"]

C["x(t)"]

A --> B --> C
```

Gradient path:

```text
shorter
↓
easier optimization
↓
faster convergence
```

---

# 8. Connection to Recurrence

```mermaid
flowchart LR

P["Previous Token"]

C["Current Token"]

M["Time-Mixing"]

H["Transformer"]

P --> M
C --> M
M --> H
```

Approximation:

$$
h_t= f(x_t,h_{t-1})
$$

without introducing recurrent computation.

---

# 9. Relationship with RWKV

```mermaid
flowchart LR

A["Shifted Tokens"]

B["Hard Time Mixing"]

C["RWKV"]

D["Learnable Time Mixing"]

A --> B
B --> C
C --> D
```

RWKV:

$$
x_k= x_t \odot \mu + x_{t-1}\odot(1-\mu)
$$

Shifted Tokens:

$$
\mu \in {0,1}
$$

---

# 10. Computational Cost

```mermaid
flowchart TB

A["Shift Operation"]

A --> B["Tensor Slice"]

B --> C["Concatenation"]

C --> D["No Extra Parameters"]

C --> E["No Extra Attention Cost"]

C --> F["Fully Parallel"]
```

---

# 11. Empirical Findings

```mermaid
flowchart TB

S["Shifted Tokens"]

S --> A["Character-level LM<br/>Large Improvement"]

S --> B["BPE Tokenization<br/>Small Improvement"]

S --> C["BPE + RoPE<br/>Almost No Gain"]

S --> D["Shift > 1<br/>Dimension Bottleneck"]
```

---

# 12. Complete Overview

```mermaid
flowchart TD

I["Input Tokens"]

TS["Shifted Tokens<br/>Time-Mixing"]

TB["Temporal Inductive Bias"]

ATTN["Self-Attention"]

FFN["Feed Forward"]

REC["Implicit Recurrence"]

FAST["Faster Convergence"]

OUT["Transformer Output"]

I --> TS
TS --> TB
TS --> ATTN
TS --> FFN
TS --> REC

TB --> FAST
REC --> FAST

ATTN --> OUT
FFN --> OUT
FAST --> OUT
```

---

# Key Takeaways

```text
Shifted Tokens
        │
        ├── No additional parameters
        ├── No additional FLOPs
        ├── Fully parallel
        ├── Injects temporal bias
        ├── Approximates recurrence
        ├── Improves convergence
        └── Foundation of modern Time-Mixing architectures
```
