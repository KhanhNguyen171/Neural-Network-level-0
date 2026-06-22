# ALiBi Positional Embedding - Overview

---

# 1. Big Picture

```mermaid
flowchart TB

A["Input Tokens"]

A --> B["Token Embedding"]

B --> C["Linear Projection<br/>Q, K, V"]

C --> D["Attention Logits<br/>QK^T / sqrt(d)"]

E["ALiBi Bias<br/>Linear Distance Bias"]

E --> F["Add Bias"]

D --> F

F --> G["Softmax"]

G --> H["Attention Output"]

H --> I["Feed Forward"]

I --> J["Transformer Block Output"]
```

---

# 2. Core Mathematical Idea

Traditional Attention:

```math
A_{ij} = \frac{Q_iK_j^T}{\sqrt d}
```

ALiBi:

```math
A_{ij} = \frac{Q_iK_j^T}{\sqrt d} - m_h(i-j)
```

where:

```math
m_h>0
```

and:

```math
i-j
```

is the relative distance.

---

# 3. Attention Matrix Construction

```mermaid
flowchart LR

A["Distance Matrix<br/>distance(i,j)"]

A --> B["Multiply by Head Slope"]

B --> C["Linear Bias Matrix"]

C --> D["Add to Attention Logits"]

D --> E["Softmax Attention"]
```

```math
b_{ij}=-m_h(i-j)
```

---

# 4. Distance Penalty

```text
Distance:

0   1   2   3   4   5
│   │   │   │   │   │
▼   ▼   ▼   ▼   ▼   ▼

Bias:

0  -1  -2  -3  -4  -5
```

Therefore:

```text
distance ↑
      ↓
negative bias ↑
      ↓
attention score ↓
```

---

# 5. Causal ALiBi Matrix

```text
┌──────────────────────────┐
│  0  -1  -2  -3  -4  -5  │
│  0   0  -1  -2  -3  -4  │
│  0   0   0  -1  -2  -3  │
│  0   0   0   0  -1  -2  │
│  0   0   0   0   0  -1  │
│  0   0   0   0   0   0  │
└──────────────────────────┘
```

Farther tokens receive larger penalties.

---

# 6. Inductive Bias of ALiBi

```mermaid
flowchart TB

A[Language Sequences]

A --> B[Nearby Tokens Are Highly Correlated]

A --> C[Long Dependencies Are Rare]

B --> D[Locality Prior]

C --> D

D --> E[Linear Attention Bias]

E --> F[ALiBi]
```

---

# 7. Multi-Head Behavior

```mermaid
flowchart LR

H1[Head 1<br/>Small Slope]

H2[Head 2]

H3[Head 3]

H4[Head N<br/>Large Slope]

H1 --> G1[Global Dependencies]

H2 --> G2[Medium Range]

H3 --> G3[Short Range]

H4 --> G4[Local Dependencies]
```

---

# 8. Hierarchical Receptive Fields

```text
Head 1:
──────────────────────────────
Very long context

Head 2:
───────────────────
Medium-long context

Head 3:
────────────
Medium context

Head N:
──────
Local context
```

---

# 9. Length Extrapolation Mechanism

```mermaid
flowchart TB

A[Train Sequence Length<br/>1024]

B[Test Sequence Length<br/>4096]

A --> C[Absolute PE<br/>Fails]

A --> D[ALiBi<br/>Still Valid]

D --> E[Length Extrapolation]
```

Because:

```math
b_{ij} = -m_h(i-j)
```

depends only on distance and not on:

```math
L_{train}
```

---

# 10. Why ALiBi Works

```mermaid
flowchart TB

A["Distance d"]

A --> B["Linear Penalty"]

B --> C["Attention Logit"]

C --> D["Softmax"]

D --> E["Attention Probability"]

E --> F["Exponential Distance Decay"]
```

Thus:

```math
P(attend) \propto e^{-m_h d}
```

which behaves like an exponential decay kernel.

---

# 11. Comparison with Other Position Methods

```mermaid
flowchart LR

ABS[Absolute PE]

ROPE[RoPE]

ALIBI[ALiBi]

DPB[Dynamic Position Bias]

ABS --> A1[Poor Extrapolation]

ROPE --> A2[Good Extrapolation]

ALIBI --> A3[Very Good Extrapolation]

DPB --> A4[Very Good Extrapolation]
```

---

# 12. Advantages

```mermaid
mindmap
root((ALiBi))
    No Extra Parameters
    Relative Position Information
    Length Extrapolation
    Simple Implementation
    Low Computational Cost
    Strong Locality Prior
    Compatible With Large LLMs
```

---

# 13. Limitations

```mermaid
flowchart TB

A[Large Slopes]

A --> B[Strong Local Attention]

B --> C[Long Distance Attention Vanishes]

C --> D[Weak Global Information Flow]
```

---

# 14. ALiBi in x-transformers

```mermaid
flowchart LR

A[8 Attention Heads]

A --> B[4 ALiBi Heads]

A --> C[4 Normal Heads]

B --> D[Local Modeling]

C --> E[Global Modeling]

D --> F[Hybrid Attention]

E --> F
```

---

# 15. Complete Overview

```mermaid
flowchart TB

A["Input Tokens"]

A --> B["Q K V Projection"]

B --> C["Attention Scores"]

D["ALiBi Linear Bias"]

D --> E["Add Bias"]

C --> E

E --> F["Softmax"]

F --> G["Attention Output"]

G --> H["Local and Global Dependencies"]

H --> I["Length Extrapolation"]

I --> J["Long Context Transformer"]
```

---

# Overview

```mermaid
flowchart TB

subgraph POSITION
A["Distance Between Tokens"]
B["Linear Bias"]
C["Bias Added to Attention"]
A --> B
B --> C
end

subgraph ATTENTION
D["Q K V Projection"]
E["Attention Scores"]
F["Softmax"]
G["Attention Output"]

D --> E
E --> F
F --> G
end

C --> E

subgraph EFFECT
H["Locality Prior"]
I["Multi-Scale Heads"]
J["Length Extrapolation"]

G --> H
G --> I
G --> J
end
```

---

# Summary

```text
ALiBi
│
├── No positional embeddings
├── Add linear bias to attention scores
├── Multi-scale attention heads
├── Strong locality inductive bias
├── No extra parameters
├── O(L²) complexity unchanged
├── Supports length extrapolation
└── Widely used in modern x-transformers and LLMs
```


```mermaid
mindmap
root((ALiBi))

    Core Idea
        Linear Bias
        Relative Distance
        No Position Embedding

    Mathematics
        Attention Plus Bias
        Distance Penalty
        Exponential Decay

    Properties
        No Extra Parameters
        O(L^2)
        Simple Implementation

    Inductive Bias
        Local Attention
        Multi Scale Heads

    Advantages
        Length Extrapolation
        Long Context Modeling
        Efficient Training

    Limitations
        Over Locality
        Weak Long Range Attention
```