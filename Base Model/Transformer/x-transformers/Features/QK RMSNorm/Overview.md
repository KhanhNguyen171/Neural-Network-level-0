# QK RMSNorm – Big Picture

```mermaid
flowchart LR

X[Input Tokens X]

X --> Q[Linear Projection WQ]
X --> K[Linear Projection WK]
X --> V[Linear Projection WV]

Q --> QRMS[RMSNorm on Query]
K --> KRMS[RMSNorm on Key]

QRMS --> QS[Learned Scale gq]
KRMS --> KS[Learned Scale gk]

QS --> DOT[Similarity Computation]
KS --> DOT

DOT --> SM[Softmax]
SM --> ATTN[Attention Matrix]

ATTN --> MUL[Multiply with V]
V --> MUL

MUL --> OUT[Output]

style QRMS fill:#d9f2ff
style KRMS fill:#d9f2ff
style DOT fill:#ffe6cc
style ATTN fill:#fff2cc
style OUT fill:#d5f5d5
```

# Mathematical Pipeline

```mermaid
flowchart TD

A["Q = XWQ"]
B["K = XWK"]
C["V = XWV"]

A --> D["Q̂ = Q / RMS(Q)"]
B --> E["K̂ = K / RMS(K)"]

D --> F["Q̃ = gq ⊙ Q̂"]
E --> G["K̃ = gk ⊙ K̂"]

F --> H["L = s · Q̃K̃ᵀ"]
G --> H

H --> I["A = Softmax(L)"]

I --> J["Y = AV"]
C --> J
```

# Geometric Interpretation
QK RMSNorm chuyển Attention từ phụ thuộc vào độ lớn vector sang phụ thuộc vào hướng của vector.

```mermaid
flowchart LR

A["Dot Product Attention"]

A --> B["Similarity = ||Q|| ||K|| cos(θ)"]

B --> C["Norm Explosion"]
B --> D["Softmax Saturation"]
B --> E["Training Instability"]

F["QK RMSNorm"]

F --> G["Similarity ≈ cos(θ)"]

G --> H["Bounded Logits"]
G --> I["Stable Gradients"]
G --> J["Better Scaling"]
```

# Attention Logits Stabilization
```mermaid
flowchart TB

A["Large ||Q|| or ||K||"]
A --> B["Huge Attention Logits"]

B --> C["Softmax becomes One-Hot"]

C --> D["Entropy Collapse"]
C --> E["Vanishing Gradient"]
C --> F["Dead Attention Heads"]

G["QK RMSNorm"]

G --> H["Normalize Query"]
G --> I["Normalize Key"]

H --> J["Bounded Logits"]
I --> J

J --> K["Healthy Attention Distribution"]
```

# Relationship with Cosine Attention
```mermaid
flowchart LR

A["Standard Attention"]

A --> B["QKᵀ / √d"]

C["Cosine Attention"]

C --> D["QKᵀ / (||Q|| ||K||)"]

E["QK RMSNorm"]

E --> F["Q̂K̂ᵀ"]

F --> G["Approximate Cosine Similarity"]

style E fill:#d9f2ff
style G fill:#d5f5d5
```

# Scaling Law Perspective
```mermaid
flowchart TD

A["Increase Model Size"]

A --> B["Larger Hidden Dimension"]
A --> C["More Layers"]
A --> D["More Parameters"]

B --> E["Norm Growth"]
C --> E
D --> E

E --> F["Attention Instability"]

G["QK RMSNorm"]

G --> H["Control Logit Magnitude"]

H --> I["Stable Optimization"]

I --> J["Train 10B+ Models"]
I --> K["Less LR Tuning"]
I --> L["Better Final Performance"]
```

# Complete Transformer Block with QK RMSNorm
```mermaid
flowchart TB

X["Input"]

X --> LN1["Pre-Norm"]

LN1 --> Q["WQ"]
LN1 --> K["WK"]
LN1 --> V["WV"]

Q --> QRMS["Q RMSNorm"]
K --> KRMS["K RMSNorm"]

QRMS --> QS["Scale gq"]
KRMS --> KS["Scale gk"]

QS --> ATTN["Scaled Dot Product Attention"]
KS --> ATTN
V --> ATTN

ATTN --> PROJ["Output Projection"]

PROJ --> ADD1["Residual Add"]

ADD1 --> FFN["Feed Forward"]

FFN --> ADD2["Residual Add"]

ADD2 --> Y["Output"]
```

# Core Formula
```math
\hat Q = g_q \odot \frac {Q} {\sqrt{ \frac {1} {d} \sum_i Q_i^2 + \epsilon}}
```

```math
\hat K = g_k \odot \frac {K} {\sqrt{ \frac {1} {d} \sum_i K_i^2 + \epsilon}}
```

```math
\text{Attention} (Q, K, V) = \text{Softmax} \left( s \cdot \hat Q \hat K^T \right)V
```

# One-Page Overview
```mermaid
mindmap
  root((QK RMSNorm))

    Motivation
      Attention instability
      Logit explosion
      Softmax saturation
      Learning rate sensitivity

    Method
      Normalize Query
      Normalize Key
      Learned Scale
      Similarity Rescaling

    Mathematics
      RMSNorm
      Approximate Cosine Similarity
      Bounded Logits

    Benefits
      Stable gradients
      Higher entropy
      Better scaling laws
      Easier optimization
      Billion-parameter training

    Applications
      x-transformers
      Persimmon-8B
      Large Language Models
      Deep Transformers
      Cosine Attention
```

