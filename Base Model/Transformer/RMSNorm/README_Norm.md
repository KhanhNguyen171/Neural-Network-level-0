# Normalization in Deep Learning

## Từ BatchNorm đến GroupNorm và RMSNorm

---

# 1. Tại sao cần Normalization?

Trong mạng neural sâu, tín hiệu được truyền qua rất nhiều lớp:

$$
x
\rightarrow
f_1
\rightarrow
f_2
\rightarrow
...
\rightarrow
f_L
$$

Sau mỗi phép biến đổi tuyến tính:

$$
y = Wx+b
$$

độ lớn của activation có thể thay đổi đáng kể.

Nếu activation quá lớn:

$$
||x|| \gg 1
$$

gradient dễ bùng nổ.

Nếu activation quá nhỏ:

$$
||x|| \ll 1
$$

gradient dễ biến mất.

Mục tiêu của Normalization là:

* Giữ activation ở một miền ổn định.
* Giảm độ nhạy với scale.
* Tăng tốc hội tụ.
* Cải thiện khả năng tối ưu.

---

# 2. Batch Normalization

## Ý tưởng

BatchNorm chuẩn hóa theo chiều batch.

Cho batch:

$$
B={x_1,x_2,\ldots,x_m}
$$

Tính mean:

$$
\mu_B =

\frac1m
\sum_{i=1}^{m}
x_i
$$

Tính variance:

$$
\sigma_B^2 =

\frac1m
\sum_{i=1}^{m}
(x_i-\mu_B)^2
$$

Chuẩn hóa:

$$
\hat{x} =

\frac{x-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
$$

Scale và Shift:

$$
y
=

\gamma \hat{x}
+
\beta
$$

---

## Tính chất

BatchNorm tạo ra:

$$
E[\hat{x}] =

0
$$

$$
Var[\hat{x}] =

1
$$

---

## Hạn chế

BN phụ thuộc batch size.

Khi:

$$
m \rightarrow nhỏ
$$

ước lượng:

$$
\mu_B,\sigma_B^2
$$

không còn chính xác.

Hiệu năng giảm mạnh với:

* Object Detection
* Segmentation
* Video Models
* Large Models bị giới hạn bộ nhớ

Đây chính là động lực cho GroupNorm.

---

# 3. Layer Normalization

LayerNorm loại bỏ hoàn toàn batch dimension.

Cho:

$$
x=[x_1,x_2,\ldots,x_d]
$$

Mean:

$$
\mu =

\frac1d
\sum_{i=1}^{d}
x_i
$$

Variance:

$$
\sigma^2=

\frac1d
\sum_{i=1}^{d}
(x_i-\mu)^2
$$

Chuẩn hóa:

$$
LN(x)=

\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

---

## Đặc điểm

LayerNorm:

* Không phụ thuộc batch size.
* Hoạt động tốt với RNN.
* Hoạt động tốt với Transformer.

Transformer gốc sử dụng LayerNorm ở mọi block.

---

# 4. Group Normalization (GN)

## Động cơ nghiên cứu

Bài báo Group Normalization nhận thấy:

> BatchNorm thất bại khi batch size nhỏ.

Ý tưởng:

Không chuẩn hóa theo batch.

Không chuẩn hóa toàn bộ feature.

Mà chia channel thành nhiều nhóm.

---

## Công thức

Giả sử tensor:

$$
X
\in
\mathbb{R}^{N\times C\times H\times W}
$$

Chia:

$$
C
\rightarrow
G
\text{ groups}
$$

Mỗi nhóm có:

$$
\frac{C}{G}
$$

channels.

Với từng group:

$$
\mu_g=

\frac1m
\sum x_i
$$

$$
\sigma_g^2=

\frac1m
\sum
(x_i-\mu_g)^2
$$

Chuẩn hóa:

$$
GN(x)=

\frac{x-\mu_g}
{\sqrt{\sigma_g^2+\epsilon}}
$$

---

## Các trường hợp đặc biệt

Nếu:

$$
G=1
$$

thì:

$$
GN
=

LN
$$

Nếu:

$$
G=C
$$

thì:

$$
GN
=

InstanceNorm
$$

GN là cầu nối giữa LN và InstanceNorm.

---

## Kết quả quan trọng

GN không phụ thuộc batch size.

Ngay cả khi:

$$
Batch=2
$$

GN vẫn ổn định hơn BN.

Đây là lý do GN trở thành chuẩn trong:

* Detection
* Segmentation
* Video Understanding

---

# 5. LayerNorm có thực sự cần Mean?

Đây là câu hỏi dẫn tới RMSNorm.

LayerNorm thực hiện:

1. Mean Centering

$$
x \rightarrow x-\mu
$$

2. Variance Scaling

$$
x \rightarrow
\frac{x}{\sigma}
$$

Nhưng liệu bước (1) có thực sự cần thiết?

Bài RMSNorm đặt giả thuyết:

> Re-centering invariance không phải yếu tố cốt lõi của LayerNorm.

---

# 6. RMSNorm

## Root Mean Square

Định nghĩa:

$$
RMS(x)=

\sqrt{
\frac1d
\sum_{i=1}^{d}
x_i^2
}
$$

---

## Chuẩn hóa

Thay vì:

$$
x-\mu
$$

RMSNorm giữ nguyên vector.

Chỉ scale:

$$
RMSNorm(x)=

\gamma
\odot
\frac{x}
{
\sqrt{
\frac1d
\sum_{i=1}^{d}
x_i^2
+\epsilon
}
}
$$

---

# 7. Ý nghĩa hình học

Norm Euclidean:

$$
||x||_2=

\sqrt{
\sum_i x_i^2
}
$$

Ta có:

$$
RMS(x)=

\frac{||x||_2}
{\sqrt d}
$$

Do đó:

$$
RMSNorm(x)
\propto
\frac{x}
{||x||_2}
$$

RMSNorm gần như đưa vector lên một hypersphere.

---

## Điều được giữ lại

Hướng vector.

## Điều bị loại bỏ

Độ lớn vector.

---

# 8. Scale Invariance

Cho:

$$
x'=cx
$$

Ta có:

$$
RMS(x')=

cRMS(x)
$$

Suy ra:

$$
\frac{x'}{RMS(x')}=

\frac{x}{RMS(x)}
$$

Nên:

$$
RMSNorm(cx)=

RMSNorm(x)
$$

Đây là tính chất re-scaling invariance của RMSNorm.

---

# 9. Partial RMSNorm

Bài báo RMSNorm còn đề xuất:

## pRMSNorm

Thay vì dùng toàn bộ vector:

$$
d
$$

chiều.

Chỉ lấy:

$$
p%
$$

số chiều để ước lượng RMS.

$$
RMS_p(x)=

\sqrt{
\frac1k
\sum_{i=1}^{k}
x_i^2
}
$$

với:

$$
k= p% \times d
$$

Ý tưởng:

* Giảm chi phí tính toán.
* Giữ nguyên tính chất scale invariance.

---

# 10. So sánh các Normalization

| Method       | Mean       | Variance | Batch Dependent |
| ------------ | ---------- | -------- | --------------- |
| BatchNorm    | Batch      | Batch    | Yes             |
| LayerNorm    | Sample     | Sample   | No              |
| InstanceNorm | Instance   | Instance | No              |
| GroupNorm    | Group      | Group    | No              |
| RMSNorm      | Không dùng | RMS      | No              |

---

# 11. Transformer và LLM hiện đại

Transformer gốc:

$$
LayerNorm
$$

BERT:

$$
LayerNorm
$$

GPT-2:

$$
LayerNorm
$$

Các LLM hiện đại:

* LLaMA
* Mistral
* Gemma
* Qwen
* DeepSeek

chuyển sang:

$$
RMSNorm
$$

vì:

* Ít phép tính hơn.
* Ít truy cập bộ nhớ hơn.
* Dễ scale hơn.
* Hiệu năng tương đương LayerNorm.

---

# 12. Kết luận

Lịch sử của Normalization có thể xem như quá trình loại bỏ dần các giả định không cần thiết:

$$
BatchNorm
\rightarrow
LayerNorm
\rightarrow
GroupNorm
\rightarrow
RMSNorm
$$

Trong đó:

* BatchNorm giải quyết ổn định huấn luyện.
* GroupNorm loại bỏ phụ thuộc batch size.
* LayerNorm phù hợp với Transformer.
* RMSNorm nhận ra rằng việc kiểm soát độ lớn quan trọng hơn việc đưa mean về 0.

Đây là lý do RMSNorm trở thành chuẩn mặc định của hầu hết kiến trúc LLM hiện đại.


# 13. Review

## Normalization Family Overview

| Loại | Động cơ | Công thức (B,C,H,W) | Đặc điểm |
|--------|----------|----------|----------|
| **BN (Batch Normalization)** | Giảm **Internal Covariate Shift**, tăng tốc hội tụ và ổn định huấn luyện | Với mỗi channel:  $$ \mu_c=\frac{1}{NHW}\sum x $$ $$ \sigma_c^2=\frac{1}{NHW}\sum(x-\mu_c)^2 $$ $$ y=\gamma\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta $$ | - Hiệu quả tốt với CNN.<br>- Tăng tốc huấn luyện.<br>- Phụ thuộc Batch Size.<br>- Kém hiệu quả khi batch nhỏ. |
| **LN (Layer Normalization)** | Loại bỏ phụ thuộc Batch Size, phù hợp cho RNN và Transformer | Với mỗi sample:  $$ \mu=\frac1d\sum_{i=1}^{d}x_i $$ $$ \sigma^2=\frac1d\sum_{i=1}^{d}(x_i-\mu)^2 $$ $$ y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta $$ | - Không phụ thuộc batch.<br>- Phù hợp RNN.<br>- Nền tảng của Transformer. |
| **IN (Instance Normalization)** | Được thiết kế cho Style Transfer và xử lý ảnh | Với từng sample và channel:  $$ \mu=\frac1{HW}\sum x $$ $$ \sigma^2=\frac1{HW}\sum(x-\mu)^2 $$ $$ y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta $$ | - Chuẩn hóa riêng từng ảnh.<br>- Loại bỏ ảnh hưởng giữa các mẫu trong batch.<br>- Phù hợp Style Transfer. |
| **GN (Group Normalization)** | Khắc phục điểm yếu của BN khi Batch Size nhỏ | Chia channel thành G nhóm:  $$ \mu_g=\frac1m\sum x_i $$ $$ \sigma_g^2=\frac1m\sum(x_i-\mu_g)^2 $$ $$ y=\gamma\frac{x-\mu_g}{\sqrt{\sigma_g^2+\epsilon}}+\beta $$ | - Không phụ thuộc Batch Size.<br>- Ổn định khi batch nhỏ.<br>- Cầu nối giữa LN và IN.<br>- Phổ biến trong Detection và Segmentation. |
| **AdaLN / FiLM (Adaptive Layer Normalization)** | Điều kiện hóa quá trình chuẩn hóa bằng thông tin bên ngoài | $$ y=\gamma(z)\odot LN(x)+\beta(z) $$ | - Scale và Shift phụ thuộc điều kiện $z$.<br>- Được dùng trong Diffusion Models và Multimodal Models. |
| **RMSNorm (Root Mean Square Normalization)** | Loại bỏ Mean-Centering, chỉ giữ Re-Scaling Invariance | $$ RMS(x)=\sqrt{\frac1d\sum_{i=1}^{d}x_i^2} $$ $$ y=\gamma\odot\frac{x}{RMS(x)+\epsilon} $$ | - Không dùng Mean.<br>- Không dùng Variance.<br>- Nhanh hơn LayerNorm.<br>- Được sử dụng trong LLaMA, Mistral, Gemma, DeepSeek. |
| **Pre-Norm** | Giải quyết vấn đề Gradient khi Transformer rất sâu | $$ y=x+F(Norm(x)) $$ | - Gradient ổn định.<br>- Huấn luyện mô hình sâu dễ hơn.<br>- Là chuẩn hiện nay của LLM. |
| **Post-Norm** | Thiết kế gốc của Transformer 2017 | $$ y=Norm(x+F(x)) $$ | - Kiến trúc đơn giản.<br>- Dễ mất ổn định khi số layer lớn. |
| **QK-Norm (Scaling Vision Transformers to 22 Billion Parameters)** | Ổn định Attention khi mô hình cực lớn | $$ Q'=\frac{Q}{\|Q\|} $$ $$ K'=\frac{K}{\|K\|} $$ $$ Attention=Q'K'^T $$ | - Giảm bão hòa Softmax.<br>- Ổn định huấn luyện ở quy mô hàng tỷ tham số. |

---

## Quan hệ giữa các phương pháp

```text
BatchNorm
    │
    ├── InstanceNorm
    │
    ├── LayerNorm
    │       │
    │       ├── RMSNorm
    │       │
    │       └── AdaLN / FiLM
    │
    └── GroupNorm
```

---

## Evolution Timeline

```text
2015 ─ BatchNorm
2016 ─ LayerNorm
2016 ─ InstanceNorm
2018 ─ GroupNorm
2019 ─ RMSNorm
2022+ ─ AdaLN
2023+ ─ QK-Norm
```