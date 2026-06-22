# NormFormer - Overview Diagram

> Visual summary of the architecture, motivations, mathematical ideas and improvements proposed by NormFormer.

---

# 1. Big Picture

```mermaid
flowchart TB

A["Deep Pre-LN Transformer"]

A --> B["Gradient Mismatch"]

B --> C["Early Layers<br/>Small Gradients"]

B --> D["Late Layers<br/>Large Gradients"]

C --> E["Slow Convergence"]
D --> E

E --> F["NormFormer"]

F --> G["Balanced Gradient Flow"]

G --> H["Faster Pretraining"]

G --> I["Improved Stability"]

G --> J["Better Scaling to Deep Models"]
```

---

# 2. Gradient Problem in Pre-LN Transformers

```mermaid
flowchart LR

L1["Layer 1"]
L2["Layer 2"]
L3["Layer 3"]
L4["Layer L"]

L1 --> G1["Small Gradient"]
L2 --> G2["Medium Gradient"]
L3 --> G3["Large Gradient"]
L4 --> G4["Very Large Gradient"]

G1 --> M["Gradient Magnitude Mismatch"]
G2 --> M
G3 --> M
G4 --> M
```

Mathematically,

```math
\left\| \nabla_{\theta_1}L \right\| \ll \left\| \nabla_{\theta_L}L \right\|
```

NormFormer aims to enforce

```math
\left\| \nabla_{\theta_1}L \right\| \approx \cdots \approx \left\| \nabla_{\theta_L}L \right\|
```

---

# 3. The Four Improvements of NormFormer

```mermaid
flowchart TB

A["NormFormer"]

A --> B["1. Per-Head Scaling"]

A --> C["2. Post-Attention LayerNorm"]

A --> D["3. Post-Activation LayerNorm"]

A --> E["4. Residual Scaling"]

B --> B1["Balance Attention Head Magnitudes"]

C --> C1["Stabilize Attention Output"]

D --> D1["Normalize FFN Activations"]

E --> E1["Control Residual Dominance"]
```

---

# 4. Per-Head Attention Scaling

```mermaid
flowchart LR

A["Head 1"] --> D["Scaled Head 1"]
B["Head 2"] --> E["Scaled Head 2"]
C["Head h"] --> F["Scaled Head h"]

D --> G["Concatenate"]
E --> G
F --> G
```

Mathematically,

```math
head_i' = g_i \cdot head_i
```

where

```math
g_i
```

is a learnable scalar.

---

# 5. Post-Attention LayerNorm

```mermaid
flowchart LR

A["Attention"]

A --> B["Output Projection"]

B --> C["LayerNorm"]

C --> D["Residual Add"]
```

Instead of

```math
x + W_OH
```

NormFormer uses

```math
x + LN(W_OH)
```

---

# 6. Post-Activation LayerNorm in FFN

```mermaid
flowchart LR

A["Linear"]

A --> B["Activation"]

B --> C["LayerNorm"]

C --> D["Linear"]

D --> E["Residual Add"]
```

Mathematically,

```math
FFN(x) = W_2 LN \left( \sigma(W_1x) \right)
```

---

# 7. Residual Scaling

```mermaid
flowchart LR

A["Residual Branch"]

A --> B["Learnable Scale α"]

B --> C["Residual Add"]
```

Instead of

```math
x + F(x)
```

NormFormer may use

```math
\alpha x + F(x)
```

where

```math
\alpha
```

is learnable.

---

# 8. Complete NormFormer Block

```mermaid
flowchart TB

X["Input x"]

X --> LN1["LayerNorm"]

LN1 --> ATT["Multi-Head Attention"]

ATT --> SCALE["Per-Head Scaling"]

SCALE --> LN2["Post-Attention LayerNorm"]

X --> ADD1

LN2 --> ADD1["Residual Add"]

ADD1 --> LN3["LayerNorm"]

LN3 --> FC1["Linear"]

FC1 --> ACT["Activation"]

ACT --> LN4["Post-Activation LayerNorm"]

LN4 --> FC2["Linear"]

ADD1 --> ADD2

FC2 --> ADD2["Residual Add"]

ADD2 --> OUT["Output"]
```

---

# 9. Mathematical View of a NormFormer Layer

```mermaid
flowchart TD

A["x"]

A --> B["LN"]

B --> C["Attention"]

C --> D["Head Scaling"]

D --> E["LN"]

E --> F["x + Attention"]

F --> G["LN"]

G --> H["FFN"]

H --> I["Activation LN"]

I --> J["x + FFN"]
```

Equivalent equations:

```math
z_1 = LN(x)
```

```math
a = LN \left( Attention(z_1) \right)
```

```math
x' = x+a
```

```math
f = W_2 LN \left( \sigma(W_1LN(x')) \right)
```

```math
y = x'+f
```

---

# 10. Gradient Flow After NormFormer

```mermaid
flowchart LR

A["Early Layers"]

B["Middle Layers"]

C["Late Layers"]

A --> D["Balanced Gradient"]

B --> D

C --> D

D --> E["Stable Optimization"]

E --> F["Faster Convergence"]

E --> G["Deep Transformer Training"]
```

---

# 11. Position of NormFormer in x-transformers

```mermaid
flowchart TB

T["Transformer"]

T --> P["Pre-LN"]

P --> N["NormFormer"]

N --> S["Sandwich Norm"]

N --> D["DeepNorm"]

N --> R["ResiDual"]

N --> X["x-transformers"]
```

---

# 12. One-Slide Summary

```mermaid
mindmap
  root((NormFormer))
    Motivation
      Gradient Mismatch
      Slow Convergence
      Deep Training Instability
    Improvements
      Per Head Scaling
      Post Attention LN
      Post Activation LN
      Residual Scaling
    Effects
      Balanced Gradients
      Stable Optimization
      Faster Pretraining
      Better Deep Scaling
    Legacy
      DeepNorm
      SandwichNorm
      ResiDual
      x-transformers
```

---

# Key Takeaways

* NormFormer does not change attention complexity.
* Adds only lightweight normalization and scaling.
* Significantly improves gradient balance.
* Enables more stable deep Transformer training.
* Forms an important building block of modern x-transformers.

