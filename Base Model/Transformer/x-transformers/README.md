# X-Transformers

> A Research-Oriented Transformer Framework

---

# 1. X-Transformers Là Gì?

X-Transformers là thư viện nghiên cứu Transformer được phát triển bởi [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers?utm_source=chatgpt.com).

Khác với các thư viện Transformer truyền thống, mục tiêu chính của X-Transformers không phải là cung cấp một mô hình cụ thể như:

* BERT
* GPT
* T5
* LLaMA

mà là xây dựng một **framework tổng quát** cho phép kết hợp và thử nghiệm hàng trăm biến thể Transformer khác nhau.

Có thể xem X-Transformers như:

$$
\text{Transformer Research Laboratory}
$$

thay vì:

$$
\text{Single Transformer Model}
$$

---

# 2. Mục Tiêu Của X-Transformers

Trong lịch sử Transformer, mỗi năm xuất hiện hàng trăm cải tiến:

* RoPE
* ALiBi
* Flash Attention
* RMSNorm
* SwiGLU
* Transformer-XL
* Multi Query Attention
* Grouped Query Attention
* Memory Tokens
* Dynamic Positional Bias

Mỗi kỹ thuật thường chỉ thay đổi một thành phần nhỏ trong kiến trúc Transformer.

X-Transformers cố gắng gom toàn bộ các ý tưởng đó vào một framework thống nhất.

Mục tiêu:

$$
Transformer =

\text{Composable Components}
$$

---

# 3. Những Kiến Thức Phải Học Trước

Không nên đọc X-Transformers nếu chưa hiểu:

---

## 3.1 Linear Algebra

Cần thành thạo:

### Matrix Multiplication

$$
C=AB
$$

### Dot Product

$$
a \cdot b =

\sum_i a_i b_i
$$

### Tensor

$$
X
\in
\mathbb{R}^{B \times N \times D}
$$

Trong đó:

* B: Batch
* N: Sequence Length
* D: Hidden Dimension

---

## 3.2 Probability

### Softmax

$$
Softmax(z_i) =

\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

### Cross Entropy

$$
L
=

-\sum_i y_i \log \hat y_i
$$

---

## 3.3 Neural Networks

Phải hiểu:

### Feed Forward Network

$$
FFN(x) =

W_2 \sigma(W_1x)
$$

### Residual

$$
y
=

x+f(x)
$$

### LayerNorm

$$
LN(x) =

\frac{x-\mu}{\sigma}
$$

---

## 3.4 Attention Mechanism

Đây là điều kiện bắt buộc.

Phải hiểu:

$$
Attention(Q,K,V) =

Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)
V
$$

Nếu chưa hiểu Attention thì không thể hiểu X-Transformers.

---

## 3.5 Encoder và Decoder

Phải hiểu:

### Encoder

$$
X
\rightarrow
H
$$

### Decoder

$$
(H,Y_{<t})
\rightarrow
Y
$$

vì toàn bộ framework được xây dựng từ:

* Encoder
* Decoder
* Encoder-Decoder

---

# 4. Tư Tưởng Cốt Lõi Của X-Transformers

Transformer gốc:

$$
Input
\rightarrow
Attention
\rightarrow
FFN
\rightarrow
Output
$$

X-Transformers xem mỗi thành phần như một module độc lập.

Ví dụ:

$$
Attention
\rightarrow
RoPE
\rightarrow
FlashAttention
\rightarrow
MQA
\rightarrow
GQA
$$

hoặc

$$
FFN
\rightarrow
GEGLU
\rightarrow
SwiGLU
\rightarrow
MixtureOfExperts
$$

hoặc

$$
LayerNorm
\rightarrow
RMSNorm
\rightarrow
ScaleNorm
$$

Tư tưởng chính:

$$
Transformer =

\text{Plug-and-Play Components}
$$

---

# 5. Kiến Trúc Nền Tảng Của Framework

Mọi biến thể đều bắt đầu từ:

<img src="assets/transformer_base.png">

$$
Input
\rightarrow
Embedding
\rightarrow
Transformer Layers
\rightarrow
Output
$$

Trong đó:

$$
TransformerLayer =

Attention
+
FeedForward
+
Residual
+
Normalization
$$

---

# 6. Các Họ Kiến Trúc Trong X-Transformers

Repository hỗ trợ ba loại kiến trúc chính.

---

## 6.1 Encoder-Only

$$
X
\rightarrow
Encoder
\rightarrow
H
$$

Kiểu BERT.

Đặc điểm:

* Bidirectional Attention
* Hiểu ngữ cảnh

---

## 6.2 Decoder-Only

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
...
$$

Kiểu GPT.

Đặc điểm:

* Causal Attention
* Sinh token

---

## 6.3 Encoder-Decoder

$$
Input
\rightarrow
Encoder
\rightarrow
Context
\rightarrow
Decoder
\rightarrow
Output
$$

Kiểu T5.

Đặc điểm:

* Cross Attention
* Seq2Seq

---

# 7. Những Thành Phần Quan Trọng Cần Học

Đây là các module xuất hiện liên tục trong repository.

---

# 8. Attention Variants

Transformer gốc:

$$
O=
Softmax
\left(
\frac{QK^T}
{\sqrt d}
\right)V
$$

Nhưng X-Transformers hỗ trợ nhiều biến thể.

---

## Multi Head Attention

$$
head_i =

Attention(Q_i,K_i,V_i)
$$

$$
MHA =

Concat(head_1,\ldots,head_h)
$$

---

## Cross Attention

$$
Q
\leftarrow Decoder
$$

$$
K,V
\leftarrow Encoder
$$

---

## Memory Attention

Bổ sung token bộ nhớ:

$$
M=
[m_1,m_2,\ldots,m_k]
$$

Cho phép lưu trữ thông tin dài hạn.

---

## Flash Attention

Mục tiêu:

Giảm chi phí bộ nhớ.

Attention chuẩn:

$$
O(n^2)
$$

Flash Attention tính toán theo block thay vì lưu toàn bộ ma trận attention.

---

# 9. Positional Encoding Variants

Transformer không biết thứ tự token.

Cần:

$$
X+P
$$

---

## Sinusoidal Position

$$
PE(pos,2i) =

sin
\left(
\frac{pos}{10000^{2i/d}}
\right)
$$

---

## Rotary Position Embedding (RoPE)

Thực hiện phép quay trong không gian vector.

Cho:

$$
q
\rightarrow
R(\theta)q
$$

$$
k
\rightarrow
R(\theta)k
$$

Được sử dụng trong:

* GPT-NeoX
* LLaMA
* Qwen

---

## ALiBi

Thay vì embedding vị trí:

$$
Score =

QK^T+b
$$

với:

$$
b
=

m \times distance
$$

---

## Dynamic Positional Bias

Một hàm học:

$$
f(i-j)
$$

để biểu diễn khoảng cách tương đối.

---

# 10. Normalization Variants

Transformer gốc:

$$
LayerNorm
$$

---

## RMSNorm

Không sử dụng mean.

$$
RMS(x) =

\sqrt{
\frac1d
\sum_i x_i^2
}
$$

$$
RMSNorm(x) =

\frac{x}{RMS(x)}
$$

Được dùng rộng rãi trong:

* LLaMA
* Mistral

---

# 11. Feed Forward Variants

Transformer gốc:

$$
FFN(x) =

W_2
\sigma(W_1x)
$$

---

## GEGLU

$$
GEGLU(x) =

(Wx)
\otimes
GELU(Vx)
$$

---

## SwiGLU

$$
SwiGLU(x) =

(Wx)
\otimes
Swish(Vx)
$$

Được dùng trong:

* PaLM
* LLaMA

---

# 12. Long Context Mechanisms

Một chủ đề lớn của repository.

Mục tiêu:

$$
n
\rightarrow
100k+
$$

token.

---

## Transformer-XL

Lưu trạng thái cũ:

$$
M_t =

[H_{t-1},H_t]
$$

Cho phép ghi nhớ dài hạn.

---

## Memory Tokens

Thêm các token:

$$
m_1,\ldots,m_k
$$

vào chuỗi.

Attention có thể đọc và ghi lên các token này.

---

# 13. Hướng Học Repository

Không nên đọc code theo thứ tự file.

Nên học theo kiến thức.

---

## Giai Đoạn 1

Hiểu:

* Attention
* Multi Head Attention
* Encoder
* Decoder

---

## Giai Đoạn 2

Hiểu:

* RoPE
* RMSNorm
* SwiGLU

Đây là nền tảng của LLM hiện đại.

---

## Giai Đoạn 3

Hiểu:

* Flash Attention
* Memory Tokens
* Transformer-XL

Đây là các kỹ thuật mở rộng context.

---

## Giai Đoạn 4

Hiểu:

* MQA
* GQA
* Dynamic Position Bias

Đây là các tối ưu hóa hiệu năng.

---

# 14. X-Transformers Và LLM Hiện Đại

Hầu hết LLM hiện nay có thể được mô tả như:

$$
LLM =

Transformer
+
RoPE
+
RMSNorm
+
SwiGLU
+
GQA
+
FlashAttention
$$

Điều đặc biệt là X-Transformers chứa gần như toàn bộ các thành phần này trong cùng một framework.

---

# 15. Kết Luận

X-Transformers không phải là một mô hình mới.

Nó là một tập hợp các ý tưởng Transformer hiện đại được chuẩn hóa trong một framework nghiên cứu thống nhất.

Tư duy cốt lõi khi học repository:

$$
Transformer =

Attention
+
Position
+
Normalization
+
FeedForward
+
Residual
$$

Mọi kiến trúc mới xuất hiện trong GPT, LLaMA, T5, PaLM, Gemma, Qwen hay Mistral đều chỉ là sự thay đổi hoặc mở rộng một trong năm thành phần nền tảng này.
