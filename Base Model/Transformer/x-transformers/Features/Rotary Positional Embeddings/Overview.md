# Minh họa tổng quát Rotary Positional Embeddings (RoPE)

## 1. Kiến trúc tổng thể trong Transformer

```mermaid
flowchart TB

A["Input Tokens x1...xn"] --> B["Token Embedding"]

B --> C["Linear Projection"]

C --> Q["Query Q"]
C --> K["Key K"]
C --> V["Value V"]

Q --> R["Rotary Positional Embedding"]
K --> R

R --> Qr["Rotated Query Q_rot"]
R --> Kr["Rotated Key K_rot"]

Qr --> ATT["Scaled Dot-Product Attention"]
Kr --> ATT
V --> ATT

ATT --> S["Softmax"]
S --> O["Output Representation"]
```
---

## 2. Cơ chế Rotary Positional Embedding (Rotation View)

```mermaid
flowchart LR

A["Embedding vector x"] --> B["Split into 2D pairs"]

B --> C["Each pair = (x_even, x_odd)"]

C --> D["Apply rotation R(p)"]

D --> E1["x' = x*cos(pθ) - y*sin(pθ)"]
D --> E2["y' = x*sin(pθ) + y*cos(pθ)"]

E1 --> F["Rotated vector"]
E2 --> F
```

---

## 3. Ý tưởng cốt lõi: Relative Position thông qua Rotation

```mermaid
flowchart TB

Qi["Query at position i"] --> Ri["Rotate R(i)"]
Kj["Key at position j"] --> Rj["Rotate R(j)"]

Ri --> QiR["Q_rot(i)"]
Rj --> KjR["K_rot(j)"]

QiR --> DOT["Dot Product"]
KjR --> DOT

DOT --> REL["Depends only on (i - j)"]
```

---

## 4. Tính chất toán học quan trọng

```mermaid
flowchart TB

A["Rotated Attention"] --> B["(R(i)Q_i) · (R(j)K_j)"]

B --> C["= Q_i · R(j - i)K_j"]

C --> D["Relative Position Encoding"]

D --> E["No absolute position needed"]

E --> F["No learnable parameters"]

F --> G["Rotation preserves geometry"]
```

---

## 5. So sánh vị trí encoding (Overview)

```mermaid
flowchart LR

A["Absolute Positional Encoding"] --> A1["Add position embedding"]
A --> A2["Learned or sinusoidal"]
A --> A3["Weak relative modeling"]

B["Relative Bias"] --> B1["Explicit bias matrix"]
B --> B2["O(n^2) memory"]

C["RoPE"] --> C1["Rotate Q & K"]
C --> C2["Implicit relative position"]
C --> C3["No parameters"]
C --> C4["Efficient & scalable"]
```

---

## 6. RoPE trong X-Transformers (thực tế triển khai)

```mermaid
flowchart TB

A["Transformer Wrapper"] --> B["Decoder Blocks"]

B --> C["Attention Layer"]

C --> D["Apply RoPE"]

D --> E["Optional: RoPE + XPos"]

E --> F["Dot-Product Attention"]

F --> G["Output Features"]
```

---

## 7. Mở rộng: RoPE + XPos (Long Context Fix)

```mermaid
flowchart TB

A["RoPE Attention"] --> B["Relative rotation encoding"]

B --> C["Limitation: poor extrapolation"]

C --> D["XPos Extension"]

D --> E["Add exponential decay bias"]

E --> F["Attention = QK - α|i-j|"]

F --> G["Better long-context performance"]
```

---

## 8. Tổng kết trực quan

```mermaid
flowchart TB

A["RoPE Core Idea"] --> B["Rotate Q and K vectors"]

B --> C["Encode position geometrically"]

C --> D["Relative position emerges naturally"]

D --> E["No additional parameters"]

E --> F["Strong inductive bias"]

F --> G["Foundation of modern LLMs"]
```
