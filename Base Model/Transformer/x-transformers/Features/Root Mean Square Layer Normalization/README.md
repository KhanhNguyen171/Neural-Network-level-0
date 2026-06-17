# Root Mean Square Layer Normalization (RMSNorm)

> A Simplified and Scalable Alternative to Layer Normalization for Modern Transformers

---

# 1. Introduction

Trong kiến trúc Transformer nguyên thủy, mọi khối Attention và Feed Forward đều sử dụng **Layer Normalization (LayerNorm)** để ổn định quá trình huấn luyện.

LayerNorm chuẩn hóa vector đặc trưng bằng cách:

1. Trừ trung bình (mean centering)
2. Chuẩn hóa theo phương sai (variance normalization)
3. Áp dụng tham số học được

Tuy nhiên khi kích thước mô hình tăng lên hàng tỷ tham số, nhiều nghiên cứu phát hiện rằng:

* Mean centering không thực sự cần thiết
* Bias không đóng góp đáng kể
* LayerNorm tiêu tốn tài nguyên tính toán
* Gây thêm độ trễ trong training và inference

Điều này dẫn tới sự ra đời của:

**Root Mean Square Layer Normalization (RMSNorm)**

RMSNorm loại bỏ hoàn toàn phép trừ mean và chỉ giữ lại chuẩn hóa độ lớn vector.

Paper gốc:

> Root Mean Square Layer Normalization
> Zhang & Sennrich (2019)

---

# 2. Motivation

## LayerNorm

Cho vector đầu vào:

$$
x = [x_1,x_2,\ldots,x_d]
$$

LayerNorm tính:

### Mean

$$
\mu = \frac1d \sum_{i=1}^{d}x_i
$$

### Variance

$$
\sigma^2= \frac1d \sum_{i=1}^{d}(x_i-\mu)^2
$$

### Normalize

$$
y_i= \gamma_i \frac{x_i-\mu} {\sqrt{\sigma^2+\epsilon}} + \beta_i
$$

Cần:

* tính mean
* tính variance
* learned scale
* learned bias

Tổng cộng nhiều phép toán phụ.

---

## Observation

Transformer thực chất quan tâm tới:

* hướng vector (direction)
* tương quan giữa các token

không nhất thiết cần:

$$
\mathbb E[x]=0
$$

Do đó:

> Chỉ cần kiểm soát độ lớn vector.

---

# 3. Root Mean Square

RMSNorm sử dụng:

$$
RMS(x)= \sqrt{ \frac1d \sum_{i=1}^{d} x_i^2 }
$$

đây là chuẩn L2 trung bình của vector.

---

## RMSNorm

Thay vì:

$$
x-\mu
$$

ta giữ nguyên:

$$
x
$$

và chuẩn hóa bằng RMS:

$$
y= \gamma \frac{x} {RMS(x)+\epsilon}
$$

hay:

$$
y= \gamma \frac{x} { \sqrt{ \frac1d \sum_{i=1}^{d}x_i^2 + \epsilon} }
$$

---

# 4. RMSNorm Pipeline

```text
Input Vector x
       │
       ▼
Compute x²
       │
       ▼
Mean(x²)
       │
       ▼
Square Root
       │
       ▼
 RMS(x)
       │
       ▼
x / RMS(x)
       │
       ▼
Multiply γ
       │
       ▼
Output
```

---

# 5. Geometric Interpretation

LayerNorm:

```text
      Mean Shift
            +
Variance Scaling
```

RMSNorm:

```text
Only Length Scaling
```

---

LayerNorm thay đổi:

* vị trí tâm
* độ dài vector

RMSNorm chỉ thay đổi:

* độ dài vector

và giữ nguyên hướng.

---

## Minh họa

```text
Before

      x
     ↗

Length = L


After RMSNorm

      y
     ↗

Length ≈ constant
Direction unchanged
```

---

# 6. Why RMSNorm Works

Attention sử dụng:

$$
QK^T
$$

Trong đó thông tin quan trọng là:

* góc giữa vector
* hướng vector

không phải giá trị trung bình.

Nếu độ lớn vector được kiểm soát:

$$
||x||
$$

thì attention vẫn ổn định.

Do đó:

```text
Mean Centering
        ↓
Less Important

Magnitude Control
        ↓
Very Important
```

---

# 7. Computational Advantage

## LayerNorm

Cần:

```text
mean
variance
subtraction
division
scale
bias
```

---

## RMSNorm

Chỉ cần:

```text
square
mean
sqrt
division
scale
```

---

Giảm:

```text
mean computation
mean subtraction
bias addition
```

---

# 8. Training Stability

RMSNorm giúp:

```text
Gradient Explosion ↓
Gradient Vanishing ↓
Training Stability ↑
```

vì mọi vector đều có chuẩn gần như nhau:

$$
||x|| \approx constant
$$

---

# 9. RMSNorm in Residual Transformers

Khối Transformer hiện đại:

```text
x
│
├──────────────┐
│              │
▼              │
RMSNorm        │
│              │
▼              │
Attention      │
│              │
▼              │
Add Residual ◄─┘
```

---

Pre-Norm Transformer:

```text
x
│
▼
RMSNorm
│
▼
Attention
│
▼
Residual Add
```

Đây là cấu hình đang được sử dụng rộng rãi.

---

# 10. RMSNorm inside x-transformers

Trong x-transformers:

```python
Decoder(
    dim = 512,
    depth = 6,
    heads = 8,
    use_rmsnorm = True
)
```

Mọi LayerNorm được thay bằng RMSNorm.

---

Pipeline:

```text
Input
 │
 ▼
RMSNorm
 │
 ▼
Multi Head Attention
 │
 ▼
Residual
 │
 ▼
RMSNorm
 │
 ▼
Feed Forward
 │
 ▼
Residual
```

---

# 11. Simple RMSNorm

Nghiên cứu năm 2023 chỉ ra rằng:

Tham số scale học được:

$$
\gamma
$$

gần như không đóng góp hiệu năng.

Do đó có thể bỏ hoàn toàn.

---

## Simple RMSNorm

$$
y= \frac{x} {RMS(x)}
$$

---

hay:

$$
y= \sqrt d \frac{x} {||x||_2}
$$

---

Pipeline:

```text
Input
 │
 ▼
L2 Norm
 │
 ▼
Scale by √d
 │
 ▼
Output
```

---

# 12. Relationship with L2 Normalization

Ta có:

$$
||x||_2= \sqrt{ \sum_i x_i^2 }
$$

và:

$$
RMS(x)=\frac{||x||_2}{\sqrt d}
$$

nên:

$$
\frac{x}{RMS(x)}= \sqrt d \frac{x}{||x||_2}
$$

RMSNorm thực chất là:

> Dimension-scaled L2 normalization.

---

# 13. RMSNorm in Modern LLMs

## Gopher

DeepMind thay LayerNorm bằng RMSNorm.

---

## RETRO

DeepMind sử dụng RMSNorm trên toàn bộ kiến trúc retrieval transformer.

---

## PaLM

Google áp dụng RMSNorm trong các khối Transformer quy mô lớn.

---

## LLaMA

Meta sử dụng RMSNorm làm chuẩn mặc định.

---

## Mistral

Tiếp tục kế thừa thiết kế RMSNorm.

---

## x-transformers

Hỗ trợ:

```python
use_rmsnorm = True
```

và

```python
use_simple_rmsnorm = True
```

---

# 14. Comparison

| Property               | LayerNorm | RMSNorm   |
| ---------------------- | --------- | --------- |
| Mean Centering         | Yes       | No        |
| Variance Normalization | Yes       | No        |
| RMS Scaling            | No        | Yes       |
| Learned Scale          | Yes       | Yes       |
| Learned Bias           | Yes       | No        |
| FLOPs                  | Higher    | Lower     |
| Memory Access          | Higher    | Lower     |
| Training Stability     | High      | High      |
| LLM Adoption           | Moderate  | Very High |

---

# 15. RMSNorm Evolution

```text
BatchNorm
     │
     ▼
LayerNorm
     │
     ▼
RMSNorm
     │
     ▼
Simple RMSNorm
     │
     ▼
Pure L2 Normalization
```

---

# 16. Why Modern LLMs Prefer RMSNorm

Ba lý do chính:

### Simplicity

Ít phép toán hơn.

---

### Better Hardware Efficiency

Giảm truy cập bộ nhớ và kernel operations.

---

### Equal or Better Performance

Nhiều nghiên cứu cho thấy:

```text
LayerNorm ≈ RMSNorm
```

và trong nhiều mô hình lớn:

```text
RMSNorm > LayerNorm
```

về tốc độ huấn luyện và khả năng mở rộng.

---

# 17. Summary

RMSNorm là phiên bản tối giản của LayerNorm dựa trên giả thuyết rằng:

> Điều quan trọng trong Transformer không phải mean centering mà là kiểm soát độ lớn vector.

Thay vì:

$$
x-\mu
$$

RMSNorm chỉ sử dụng:

$$
RMS(x)= \sqrt{ \frac1d \sum x_i^2 }
$$

để chuẩn hóa.

Điều này:

* giảm chi phí tính toán
* giảm truy cập bộ nhớ
* ổn định gradient
* cải thiện khả năng mở rộng

và trở thành lựa chọn mặc định trong phần lớn các LLM hiện đại như:

* Gopher
* RETRO
* PaLM
* LLaMA
* Mistral
* x-transformers

RMSNorm hiện được xem là chuẩn normalization chủ đạo cho các kiến trúc Transformer quy mô lớn.

# 18. Tài liệu tham khảo
1. Zhang, B. & Sennrich, R. (2019). _Root Mean Square Layer Normalization_. arXiv:1910.07467.
2. _Transformers without Tears: Improving the Normalization of Self-Attention_. arXiv:1910.05895.
3. _On Layer Normalization in the Transformer Architecture_. arXiv:2002.04745.
4. _Scalable Transformers for Language Modeling_. arXiv:2102.11972.
5. _RETRO: Improving Language Models with Retrieval_. arXiv:2112.11446.
6. _The Hedgehog & the Porcupine: Removing Learned Scale in RMSNorm_. arXiv:2307.14995.
7. x-transformers implementation: https://github.com/lucidrains/x-transformers
8. DeepMind Research: https://deepmind.google/

# 19. Thư viện tham khảo
```Python
import torch
from x_transformers import TransformerWrapper, Decoder, Encoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        use_rmsnorm = True # set to true to use for all layers
    )
)
```