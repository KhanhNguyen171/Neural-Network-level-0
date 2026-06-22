# Dynamic Positional Bias (DPB) - Overview

> Học một hàm liên tục của khoảng cách tương đối nhằm cải thiện khả năng ngoại suy chiều dài (Length Extrapolation) của Transformer.

---

# 1. Big Picture

```mermaid
flowchart TB

A["Input Tokens"]
A --> B["Token Embedding"]
B --> C["Transformer Attention"]

C --> D["Compute Relative Distance<br/>d = i - j"]

D --> E["Dynamic Positional Bias Network<br/>f(d)"]

E --> F["Position Bias Matrix B"]

C --> G["Attention Scores<br/>QK^T / sqrt(d_k)"]

F --> H["Add Relative Bias"]

G --> H

H --> I["Softmax"]

I --> J["Weighted Sum of Values"]

J --> K["Transformer Output"]
```

---

# 2. Motivation

```mermaid
flowchart LR

A["Learned Relative Bias Table"]
--> B["Only defined for training length"]

B --> C["Cannot represent unseen distances"]

C --> D["Poor Length Extrapolation"]

E["Dynamic Positional Bias"]
--> F["Learn continuous function f(d)"]

F --> G["Defined for arbitrary distances"]

G --> H["Generalize to longer sequences"]
```

---

# 3. Relative Bias vs Dynamic Bias

```mermaid
flowchart TB

subgraph Traditional Relative Bias
A1["Distance d"]
A1 --> B1["Lookup Table"]
B1 --> C1["Bias b(d)"]
end

subgraph Dynamic Positional Bias
A2["Distance d"]
A2 --> B2["MLP"]
B2 --> C2["Bias f(d)"]
end
```

---

# 4. Dynamic Bias Network

```mermaid
flowchart LR

A["Relative Distance d"]

A --> B["Linear Distance<br/>x = d"]

A --> C["Log Distance<br/>x = sign(d) log(1+|d|)"]

B --> D["MLP"]
C --> D

D --> E["Linear"]
E --> F["GELU"]
F --> G["Linear"]
G --> H["GELU"]
H --> I["Linear"]

I --> J["Bias for each Head"]
```

---

# 5. Mathematical Pipeline

```text
Relative Distance
        d = i - j
             |
             v
       Distance Encoding
             |
             v
      x = d
      or
      x = sign(d) log(1+|d|)
             |
             v
           MLP
             |
             v
        b(d)=f(d)
             |
             v
Attention Logits:
QK^T / sqrt(d_k) + b(d)
             |
             v
          Softmax
             |
             v
        Attention Output
```

---

# 6. Attention Architecture

```mermaid
flowchart TB

Q["Queries Q"]
K["Keys K"]
V["Values V"]

Q --> S["QK^T"]

K --> S

D["Relative Distance Matrix"]
--> M["Dynamic Bias MLP"]

M --> B["Bias Matrix"]

S --> A["Attention Logits"]

B --> A

A --> P["Softmax"]

P --> O["Attention Weights"]

O --> R["Weighted Values"]

V --> R

R --> T["Output"]
```

---

# 7. Length Extrapolation

```mermaid
flowchart LR

A["Training Length = 128"]
--> B["Learn f(d)"]

B --> C["Distance = 256"]

B --> D["Distance = 512"]

B --> E["Distance = 1024"]

B --> F["Distance = 4096"]

C --> G["Inference"]

D --> G

E --> G

F --> G
```

---

# 8. Linear vs Log Distance

```mermaid
flowchart TB

A["Relative Distance"]

A --> B["Linear Distance"]

A --> C["Log Distance"]

B --> D["Language Models"]

B --> E["Autoregressive Models"]

C --> F["Vision Transformers"]

C --> G["Resolution Extrapolation"]
```

---

# 9. Comparison with Other Positional Methods

```mermaid
flowchart TB

A["Positional Encoding Methods"]

A --> B["Absolute PE"]
A --> C["Relative Bias"]
A --> D["RoPE"]
A --> E["ALiBi"]
A --> F["Dynamic Positional Bias"]

B --> B1["Poor Extrapolation"]

C --> C1["Bounded by Table Size"]

D --> D1["Moderate Extrapolation"]

E --> E1["Excellent Extrapolation"]

F --> F1["Learned Continuous Function"]
```

---

# 10. DPB inside x-Transformers

```mermaid
flowchart TB

A["Input Tokens"]

A --> B["Embedding"]

B --> C["Multi-Head Attention"]

C --> D["Dynamic Positional Bias"]

D --> E["Attention Logits"]

E --> F["Softmax"]

F --> G["Transformer Layer Output"]

G --> H["Stacked Transformer Layers"]
```

---

# 11. Scientific Interpretation

```mermaid
mindmap
  root((Dynamic Positional Bias))

    Continuous Function
      Learn f(d)
      Smooth Geometry
      Relative Distance

    Generalization
      Long Context
      Length Extrapolation
      Unseen Distances

    Efficiency
      Small MLP
      Few Parameters
      Architecture Independent

    Applications
      Language Models
      Vision Transformers
      RNA Folding
      Long Sequence Modeling
```

---

# 12. Key Takeaway

```text
Traditional Relative Bias
        |
        +--> Learn a finite table b(d)
        |
        +--> Limited by training length

Dynamic Positional Bias
        |
        +--> Learn a continuous function f(d)
        |
        +--> Generalize to arbitrary distances
        |
        +--> Enable long-context modeling
        |
        +--> Strong positional inductive bias
```
