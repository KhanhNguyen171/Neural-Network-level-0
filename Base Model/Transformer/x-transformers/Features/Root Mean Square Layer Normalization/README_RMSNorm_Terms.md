# Appendix A — Understanding RMSNorm vs LayerNorm Properties

---

# Mục tiêu của phụ lục

Khi đọc bảng so sánh RMSNorm và LayerNorm:

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

người đọc thường gặp nhiều thuật ngữ liên quan tới toán học, tối ưu hóa và phần cứng.

Phụ lục này giải thích ý nghĩa chính xác của từng thuộc tính.

---

# 1. Mean Centering

## Mean là gì?

Cho vector:

$$
x=[x_1,x_2,...,x_d]
$$

Mean (trung bình):

$$
\mu= \frac1d \sum_{i=1}^{d}x_i
$$

---

Ví dụ hình học:

```text
Vector ban đầu

[ 2 4 6 8 ]

Mean = 5
```

---

## Mean Centering là gì?

Trừ mean khỏi mọi phần tử:

$$
x_i' = x_i - \mu
$$

---

Minh họa:

```text
Before

[ 2 4 6 8 ]

Mean = 5


After

[-3 -1 1 3]
```

---

Khi đó:

```text
Tổng các phần tử = 0
```

hay

$$
E[x]=0
$$

---

## LayerNorm

Có sử dụng mean centering.

```text
Input
  │
  ▼
Subtract Mean
  │
  ▼
Normalize
```

---

## RMSNorm

Bỏ hoàn toàn bước này.

```text
Input
  │
  ▼
Normalize Directly
```

---

# 2. Variance Normalization

## Variance là gì?

Variance đo mức độ phân tán của dữ liệu.

$$
\sigma^2= \frac1d \sum_{i=1}^{d} (x_i-\mu)^2
$$

---

Minh họa:

```text
Vector A

[5 5 5 5]

Variance = 0


Vector B

[0 0 10 10]

Variance = lớn
```

---

## Variance Normalization

Chia cho độ lệch chuẩn:

$$
\sigma= \sqrt{\sigma^2}
$$

để đưa dữ liệu về cùng thang đo.

---

Pipeline:

```text
Input
 │
 ▼
Compute Mean
 │
 ▼
Compute Variance
 │
 ▼
Divide by Std
 │
 ▼
Output
```

---

LayerNorm thực hiện bước này.

RMSNorm không thực hiện.

---

# 3. RMS Scaling

## RMS là gì?

Root Mean Square:

$$
RMS(x)= \sqrt{ \frac1d \sum_i x_i^2 }
$$

---

Pipeline:

```text
x
│
▼
Square
│
▼
Mean
│
▼
Square Root
│
▼
RMS
```

---

## Ý nghĩa

RMS đo:

```text
Độ lớn trung bình của vector
```

không quan tâm:

```text
Mean
Variance
```

---

Ví dụ:

```text
Vector

[3 4]

RMS = sqrt((9+16)/2) = 3.535
```

---

RMSNorm chuẩn hóa:

$$
x/RMS(x)
$$

---

# 4. Learned Scale (γ)

## Scale Parameter

Là tham số học được:

$$
\gamma
$$

---

Sau khi chuẩn hóa:

$$
y= \gamma \cdot x_{norm}
$$

---

Mục tiêu:

```text
Cho mô hình tự học
độ lớn phù hợp
của từng feature
```

---

Minh họa:

```text
Normalized

[1 1 1]

γ = 3

Output

[3 3 3]
```

---

## Vì sao cần?

Nếu mọi vector luôn có độ dài bằng nhau:

```text
Mô hình mất tự do biểu diễn
```

γ cho phép mạng khôi phục năng lực đó.

---

# 5. Learned Bias (β)

## Bias Parameter

Là tham số:

$$
\beta
$$

---

Output:

$$
y= \gamma x + \beta
$$

---

Minh họa:

```text
Normalized

[1 1 1]

β = 2

Output

[3 3 3]
```

---

## Vai trò

Dịch chuyển toàn bộ phân phối.

```text
Shift Distribution
```

---

LayerNorm:

```text
Uses β
```

RMSNorm:

```text
No β
```

---

# 6. FLOPs

## FLOPs là gì?

Floating Point Operations.

Số phép tính dấu phẩy động.

---

Ví dụ:

```text
1 phép cộng

a+b = 1 FLOP
```

---

```text
1 phép nhân

a*b = 1 FLOP
```

---

Trong thực tế:

```text
FLOPs ≈ chi phí tính toán
```

---

LayerNorm:

```text
Compute Mean
Compute Variance
Subtract Mean
Normalize
Scale
Bias
```

nên FLOPs cao hơn.

---

RMSNorm:

```text
Square
Mean
Sqrt
Normalize
Scale
```

ít phép toán hơn.

---

# 7. Memory Access

## Memory Access là gì?

Số lần phải đọc/ghi dữ liệu từ bộ nhớ.

---

Pipeline GPU:

```text
HBM
 │
 ▼
SRAM
 │
 ▼
Tensor Core
```

---

Mỗi lần truy cập bộ nhớ:

```text
Tốn năng lượng
Tốn thời gian
```

---

LayerNorm:

```text
Read x
Read mean
Read variance
Read gamma
Read beta
```

---

RMSNorm:

```text
Read x
Read gamma
```

---

Do đó:

```text
Memory Access thấp hơn
```

---

# 8. Training Stability

## Training Stability là gì?

Khả năng mô hình học ổn định.

---

Mô hình ổn định khi:

```text
Gradient không nổ
Gradient không biến mất
Loss giảm đều
```

---

Minh họa:

```text
Stable

Loss
 │
 │\
 │ \
 │  \
 │   \
 └──────► Step
```

---

Không ổn định:

```text
Loss
 │\/\ /\__
 │ /\ /
 │/
 └──────► Step
```

---

Mục tiêu của normalization:

```text
Ổn định độ lớn activation
```

để gradient ổn định hơn.

---

# 9. LLM Adoption

## Adoption là gì?

Mức độ được sử dụng trong các mô hình thực tế.

---

## LayerNorm

Được dùng trong:

```text
Transformer
BERT
GPT-2
T5
```

---

## RMSNorm

Được dùng trong:

```text
Gopher
RETRO
PaLM
LLaMA
Mistral
Mixtral
Gemma
DeepSeek
Qwen
```

---

Minh họa tiến hóa:

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
```

---

# Tổng kết

```text
Mean Centering
    ↓
Đưa trung bình về 0

Variance Normalization
    ↓
Đưa độ phân tán về chuẩn

RMS Scaling
    ↓
Chuẩn hóa độ lớn vector

Learned Scale (γ)
    ↓
Điều chỉnh độ lớn đầu ra

Learned Bias (β)
    ↓
Dịch chuyển đầu ra

FLOPs
    ↓
Chi phí tính toán

Memory Access
    ↓
Chi phí đọc ghi bộ nhớ

Training Stability
    ↓
Độ ổn định khi huấn luyện

LLM Adoption
    ↓
Mức độ sử dụng trong LLM hiện đại
```

RMSNorm có thể được xem là:

```text
LayerNorm
    ├─ bỏ Mean Centering
    ├─ bỏ Variance Normalization
    ├─ bỏ Bias
    └─ giữ lại kiểm soát độ lớn vector
```

Đây chính là lý do RMSNorm trở thành normalization mặc định trong phần lớn các LLM thế hệ mới.
