# Appendix: Giải Thích Các Biến Thể FFN-GLU Trong Bài Báo *GLU Variants Improve Transformer*

> Phụ lục này giải thích chi tiết công thức (6) trong bài báo của Noam Shazeer nhằm giúp người đọc hiểu bản chất toán học, cơ chế hoạt động và sự khác biệt giữa các biến thể GLU được sử dụng trong Transformer hiện đại.

---

# 1. Công Thức Gốc Trong Bài Báo

Bài báo định nghĩa các biến thể Feed Forward Network như sau:

$$
FFN_{GLU}(x,W,V,W_2)= (\sigma(xW)\otimes xV)W_2
$$

$$
FFN_{Bilinear}(x,W,V,W_2)= (xW\otimes xV)W_2
$$

$$
FFN_{ReGLU}(x,W,V,W_2)= (ReLU(xW)\otimes xV)W_2
$$

$$
FFN_{GEGLU}(x,W,V,W_2)= (GELU(xW)\otimes xV)W_2
$$

$$
FFN_{SwiGLU}(x,W,V,W_2)= (Swish(xW)\otimes xV)W_2
$$

Trong đó:

$$
\otimes
$$

là phép nhân từng phần tử (Element-wise Multiplication).

---

# 2. Ký Hiệu Trong Công Thức

## Input

Đầu vào Transformer:

$$
x \in \mathbb{R}^{d}
$$

---

## Ma trận W

$$
W
$$

tạo ra nhánh Gate.

```text
Input
  │
  ▼
 xW
```

---

## Ma trận V

$$
V
$$

tạo ra nhánh Feature.

```text
Input
  │
  ▼
 xV
```

---

## Ma trận W₂

$$
W_2
$$

chiếu kết quả trở lại kích thước mô hình.

```text
Hidden Dimension
      │
      ▼
     W₂
      │
      ▼
Model Dimension
```

---

# 3. Tư Duy Trực Quan

Mọi biến thể GLU đều có cùng một kiến trúc.

```text
                     Input x
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
          xW                      xV
            │                       │
            ▼                       ▼
      Activation                Feature
            │                       │
            └────────⊙──────────────┘
                     │
                     ▼
                   W₂
                     │
                     ▼
                  Output
```

---

## Ý nghĩa

Nhánh trái:

```text
Gate Branch
```

quyết định:

```text
Thông tin nào được phép đi qua
```

---

Nhánh phải:

```text
Feature Branch
```

chứa:

```text
Thông tin thực sự cần xử lý
```

---

# 4. Bilinear Layer

Đây là biến thể đơn giản nhất.

$$
FFN_{Bilinear}= (xW\otimes xV)W_2
$$

---

## Kiến trúc

```text
Input
 │
 ├──────────────┐
 │              │
 ▼              ▼
xW             xV
 │              │
 ▼              ▼
Feature A   Feature B
 │              │
 └──────⊙───────┘
        │
        ▼
       W₂
        │
        ▼
     Output
```

---

## Ý nghĩa

Không tồn tại Activation.

Thực chất mô hình học:

$$
Feature_A \times Feature_B
$$

---

### Ví dụ trực quan

Giả sử:

```text
Feature A = "Danh từ"

Feature B = "Động từ"
```

Bilinear cho phép mạng học:

```text
Tương tác giữa hai đặc trưng
```

thay vì từng đặc trưng độc lập.

---

# 5. GLU

## Công thức

$$
FFN_{GLU}=(\sigma(xW)\otimes xV)W_2
$$

---

## Gate

$$
\sigma(xW)
$$

là Sigmoid.

---

### Đặc tính

$$
0 \le \sigma(xW)\le 1
$$

---

## Minh họa

```text
Gate

0 ──────── 1
```

---

## Ví dụ

Giả sử:

```text
Gate = 0.9

Feature = 10
```

Output:

$$
0.9\times10=9
$$

---

Nếu:

```text
Gate = 0.1
```

Output:

$$
0.1\times10=1
$$

---

## Ý nghĩa

Gate hoạt động như:

```text
Van điều tiết
```

---

# 6. ReGLU

## Công thức

$$
FFN_{ReGLU}= (ReLU(xW)\otimes xV)W_2
$$

---

## Gate

$$
ReLU(x)=\max(0,x)
$$

---

## Minh họa

```text
Gate

        /
       /
      /
_____/
```

<p align="center">
  <img src="assets/ReLU.png" width="350">
</p>

---

## Hành vi

```text
x < 0
Gate = 0

x > 0
Gate = x
```

---

## Điểm mạnh

Không bị saturation như Sigmoid.

---

## Điểm yếu

Có thể xuất hiện:

```text
Dead Gate
```

---

# 7. GEGLU

## Công thức

$$
FFN_{GEGLU}= (GELU(xW)\otimes xV)W_2
$$

---

## Gate

$$
GELU(x)= x\Phi(x)
$$

---

## Minh họa

```text
Gate

      __
    /
  /
_/
```

<p align="center">
  <img src="assets/GELU.png" width="350">
</p>

---

## Khác biệt với ReLU

ReLU:

```text
Âm → 0
```

---

GELU:

```text
Âm → giảm dần
```

---

## Ví dụ

```text
Input = -1
```

ReLU:

```text
0
```

---

GELU:

```text
-0.16
```

Thông tin vẫn được giữ lại.

---

## Ý nghĩa

GEGLU tạo ra:

```text
Soft Gating
```

thay vì:

```text
Hard Gating
```

---

# 8. SwiGLU

## Công thức

$$
FFN_{SwiGLU}= (Swish(xW)\otimes xV)W_2
$$

---

với:

$$
Swish(x)= x\sigma(x)
$$

---

## Kiến trúc

```text
                     Input
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
           xW                      xV
            │                       │
            ▼                       ▼
         Swish                  Feature
            │                       │
            └────────⊙──────────────┘
                     │
                     ▼
                    W₂
                     │
                     ▼
                  Output
```

---

## Trực quan

```text
Gate

          __
        /
      /
____/
```

<p align="center">
  <img src="assets/Swish.png" width="350">
</p>

---

## Ý nghĩa

Swish kết hợp:

```text
Linear Region + Sigmoid Region
```

---

Do đó:

Gate vừa:

```text
Cho phép truyền thông tin
```

vừa:

```text
Điều tiết mềm
```

---

# 9. So Sánh Các Cơ Chế Gate

| Variant  | Gate     |
| -------- | -------- |
| Bilinear | Không có |
| GLU      | Sigmoid  |
| ReGLU    | ReLU     |
| GEGLU    | GELU     |
| SwiGLU   | Swish    |

---

## Trực quan

```text
Bilinear
  │
  ▼
Feature × Feature

--------------------------------

GLU
  │
  ▼
Feature × Sigmoid Gate

--------------------------------

ReGLU
  │
  ▼
Feature × ReLU Gate

--------------------------------

GEGLU
  │
  ▼
Feature × GELU Gate

--------------------------------

SwiGLU
  │
  ▼
Feature × Swish Gate
```

---

# 10. Tại Sao Có 3 Ma Trận?

FFN gốc:

$$
FFN(x)=W_2(\phi(xW_1))
$$

chỉ có:

```text
W₁
W₂
```

---

GLU:

$$
(\phi(xW)\otimes xV)W_2
$$

cần:

```text
W
V
W₂
```

---

## Minh họa

```text
FFN

Input
 │
 ▼
W₁
 │
 ▼
Activation
 │
 ▼
W₂

--------------------------------

GLU

Input
 │
 ├───────────┐
 │           │
 ▼           ▼
 W           V
 │           │
 ▼           ▼
Gate      Feature
 │           │
 └────⊙──────┘
      │
      ▼
      W₂
```

---

# 11. Vì Sao Bài Báo Giảm Hidden Dimension?

Trong bài báo có ghi:

> reduce hidden dimensions by a factor of 2/3

---

Lý do:

GLU sử dụng:

```text
W
V
W₂
```

thay vì:

```text
W₁
W₂
```

nên số tham số tăng lên.

Để giữ:

```text
Parameter Count
```

và

```text
FLOPs
```

gần như không đổi,

tác giả giảm:

$$
d_{ff} \rightarrow \frac{2}{3}d_{ff}
$$

---

## Minh họa

```text
Original FFN

d
 │
 ▼
4d
 │
 ▼
d

----------------------------

GLU FFN

d
 │
 ▼
8d/3
 │
 ▼
d
```

---

# 12. Kết Luận

Tất cả các biến thể trong bài báo đều có thể được viết dưới dạng tổng quát:

$$
FFN(x)= ( Activation(xW) \otimes xV ) W_2
$$

Trong đó:

```text
Bilinear → Identity

GLU      → Sigmoid

ReGLU    → ReLU

GEGLU    → GELU

SwiGLU   → Swish
```

Sự khác biệt duy nhất nằm ở:

```text
Activation Function
```

nhưng thay đổi nhỏ này lại tạo ra sự khác biệt lớn về:

* Gradient Flow
* Optimization Stability
* Scaling Behavior
* Language Modeling Performance

và đó là lý do GEGLU và đặc biệt là SwiGLU đã trở thành kiến trúc Feed Forward mặc định của các LLM hiện đại như PaLM, LLaMA, Gemma, Qwen, DeepSeek và Mistral.
