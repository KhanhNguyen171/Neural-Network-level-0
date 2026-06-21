# Gated Residual Connections - Overview Diagram

> Tổng quan kiến trúc, luồng thông tin, công thức toán học và vai trò của Gated Residual trong Transformer và GTrXL.

---

# 1. Big Picture

```mermaid
flowchart TB

A["Biểu diễn đầu vào x"]

A --> B["Tầng Transformer<br/>(Attention hoặc FFN)"]

B --> C["Đặc trưng biến đổi h = F(x)"]

A --> D["Đường Residual"]

A --> E["Tính toán Gate"]
C --> E

E --> F["Update Gate z"]
E --> G["Reset Gate r (tuỳ chọn)"]

D --> H["Trạng thái ứng viên h_hat"]
C --> H
G --> H

D --> I["Kết hợp có trọng số"]
F --> I
H --> I

I --> J["Đầu ra y"]

J --> K["Tầng Transformer tiếp theo"]
```

---

# 2. Information Flow

```mermaid
flowchart LR

X["Trạng thái trước x"]

F["Đặc trưng mới F(x)"]

G["Gate z"]

Y["Trạng thái cập nhật y"]

X --> G
F --> G

X --> Y
F --> Y
G --> Y
```

Residual connection trở thành:

```math
y = (1-z)\odot x + z\odot h
```

thay vì:

```math
y = x + h
```

---

# 3. Comparison with Standard Residual

```mermaid
flowchart LR

subgraph Residual tiêu chuẩn
A1["x"]
B1["F(x)"]
C1["y = x + F(x)"]

A1 --> C1
B1 --> C1
end

subgraph Gated Residual
A2["x"]
B2["F(x)"]
D2["Gate z"]
C2["y = (1-z)x + z h"]

A2 --> D2
B2 --> D2

A2 --> C2
B2 --> C2
D2 --> C2
end
```

---

# 4. Mathematical Overview

```text
Đầu vào:
    x

1. h = F(x)

2. Tính gate

       z = sigmoid(W[x, h] + b)

3. Kết hợp residual có cổng

       y = (1-z) ⊙ x
           + z ⊙ h
```

---

# 5. GRU-Style Gated Residual (GTrXL)

```mermaid
flowchart TD

A["Đầu vào x"]

A --> B["Sublayer F(x)"]
B --> C["h"]

A --> D["Reset Gate r"]
C --> D

A --> E["Update Gate z"]
C --> E

A --> F["r ⊙ x"]

F --> G["Trạng thái ứng viên h_hat"]
C --> G

G --> H["Kết hợp đầu ra"]
A --> H
E --> H

H --> I["Đầu ra y"]
```

Mathematically,

```math
r=\sigma(W_r[h,x])
```

```math
z=\sigma(W_z[h,x]-b_g)
```

```math
\hat h=\tanh(W_g(r\odot x)+U_gh)
```

```math
y=(1-z)\odot x + z\odot \hat h
```

---

# 6. Gradient Stabilization Mechanism

```mermaid
flowchart LR

A["Gradient đầu vào"]

B["Residual tiêu chuẩn"]

C["Nguy cơ bùng nổ gradient"]

D["Gated Residual"]

E["Luồng gradient ổn định"]

A --> B
B --> C

A --> D
D --> E
```

Jacobian:

```math
\frac{\partial y}{\partial x} = (1-z) + z \frac{\partial \hat h}{\partial x}
```

The gate acts as an adaptive regulator of gradient magnitude.

---

# 7. Information Filtering Perspective

```mermaid
flowchart LR

N["Nhiễu đầu vào"]

G["Gate"]

F["Thông tin đã lọc"]

N --> G
G --> F
```

Gate học được:

- giữ lại thông tin quan trọng;
- loại bỏ cập nhật nhiễu;
- điều khiển luồng đặc trưng.

---

# 8. Position inside a Transformer Block

```mermaid
flowchart TD

A["Input x"]

A --> B["LayerNorm"]

B --> C["Multi-Head Attention"]

C --> D["Gated Residual"]

D --> E["LayerNorm"]

E --> F["Feed Forward Network"]

F --> G["Gated Residual"]

G --> H["Output"]
```

---

# 9. Why Gated Residual Works

```mermaid
mindmap
  root((Gated Residual))
    Ổn định huấn luyện
      Kiểm soát Jacobian
      Giảm phương sai
      Tối ưu dễ hơn
    Định tuyến thông tin
      Giữ bộ nhớ
      Cập nhật thích nghi
      Lọc nhiễu
    Transformer sâu
      Gradient tốt hơn
      Ổn định activation
      Hội tụ nhanh hơn
    Reinforcement Learning
      Phần thưởng thưa
      Gán tín dụng dài hạn
      Dữ liệu phi dừng
```

---

# 10. Overall Summary

```mermaid
flowchart TB

A["Input x"]

A --> B["Tầng Transformer"]

B --> C["Biểu diễn mới h"]

A --> D["Tính gate z"]
C --> D

A --> E["Giữ thông tin cũ"]

C --> F["Tạo thông tin mới"]

D --> G["Kết hợp thích nghi"]

E --> G
F --> G

G --> H["Output y"]

H --> I["Transformer ổn định"]
I --> J["Transformer-XL"]
J --> K["GTrXL"]
J --> L["x-transformers"]
```

---

# Key Takeaways

```text
Standard Residual:
y = x + F(x)

Gated Residual:
y = (1-z)x + zh

GRU-Gated Residual:
y = (1-z)x + z h_hat
```

Gated Residual transforms the residual pathway from a fixed additive operation into a learnable information routing mechanism that:

✓ stabilizes optimization

✓ improves gradient propagation

✓ filters noisy updates

✓ enables deeper Transformers

✓ forms one of the core ideas behind GTrXL and modern x-transformers.
