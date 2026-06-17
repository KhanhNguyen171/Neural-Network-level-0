# No Bias in Feedforward Networks

## Tối giản hóa Feedforward Layer trong Transformer hiện đại

---

# 1. Giới thiệu

Trong các kiến trúc Transformer nguyên bản, mỗi lớp Feedforward Network (FFN) bao gồm hai phép biến đổi tuyến tính:

$$
FFN(x)=W_2 \phi(W_1x+b_1)+b_2
$$

trong đó:

* $W_1,W_2$: là ma trận trọng số
* $b_1,b_2$: là vector bias
* $\phi$: là hàm kích hoạt phi tuyến

Trong nhiều năm, bias được xem là thành phần mặc định của mạng nơ-ron. Tuy nhiên các nghiên cứu gần đây về Transformer quy mô lớn cho thấy:

> Bias trong FFN đóng góp rất ít vào năng lực biểu diễn nhưng lại làm tăng chi phí tính toán, lưu trữ và truyền dữ liệu.

Bắt đầu từ kiến trúc PaLM, một xu hướng mới xuất hiện:

> Loại bỏ hoàn toàn bias khỏi các lớp Transformer.

Điều này dẫn tới biến thể:

$$
FFN(x)=W_2 \phi(W_1x)
$$

hay còn gọi là: **Bias-Free Feedforward Network**

---

# 2. Động cơ khoa học

## 2.1 Sự dư thừa của Bias sau Layer Normalization

Trong Transformer hiện đại:

$$
x' = LN(x)
$$

đầu vào FFN thường đã trải qua:

* LayerNorm
* RMSNorm

Các cơ chế chuẩn hóa này:

* đưa trung bình về gần 0
* chuẩn hóa độ lớn vector

Do đó việc cộng thêm bias:

$$
W_1x+b_1
$$

trở nên ít ý nghĩa hơn.

Về mặt thống kê:

$$
E[LN(x)] \approx 0
$$

và:

$$
Var[LN(x)] \approx 1
$$

nên bias gần như bị hấp thụ vào quá trình học của trọng số.

---

## 2.2 Bias không tạo thêm tương tác chiều

Bias chỉ thực hiện:

$$
z_i = w_i^Tx+b_i
$$

Nó chỉ dịch chuyển hyperplane.

Trong khi đó:

$$
W_i
$$

mới là thành phần tạo ra:

* xoay không gian
* kéo giãn không gian
* tạo tương tác giữa các chiều

Do đó năng lực biểu diễn chính nằm ở trọng số.

---

## 2.3 Quy mô mô hình cực lớn

Khi:

$$
d_{model}
$$

và

$$
d_{ff}
$$

tăng lên hàng chục nghìn chiều:

$$
d_{ff} = 4d_{model}
$$

số lượng bias cũng tăng tương ứng.

Ví dụ:

$$
d_{model}=8192
$$

$$
d_{ff}=32768
$$

thì riêng FFN đã có:

$$
32768+8192
$$

tham số bias cho mỗi block.

Ở hàng trăm block Transformer:

* chi phí bộ nhớ tăng lên
* chi phí truyền dữ liệu tăng lên
* không đem lại cải thiện đáng kể về chất lượng.

---

# 3. Feedforward truyền thống

Một lớp FFN tiêu chuẩn:

$$
h=\phi(W_1x+b_1)
$$

$$
y=W_2h+b_2
$$

---

## Thuật toán

```text
Input x

z1 = W1 x + b1

h = Activation(z1)

z2 = W2 h + b2

Output = z2
```

---

## Số tham số

$$
P=d_{model} d_{ff} + d_{ff}d_{model} + d_{ff} + d_{model}
$$

hai thành phần cuối là bias.

---

# 4. Feedforward không Bias

Bias được loại bỏ hoàn toàn:

$$
h=\phi(W_1x)
$$

$$
y=W_2h
$$

---

## Thuật toán

```text
Input x

z1 = W1 x

h = Activation(z1)

z2 = W2 h

Output = z2
```

---

## Số tham số

$$
P=d_{model}d_{ff} + d_{ff}d_{model}
$$

không còn:

$$
d_{ff}+d_{model}
$$

tham số bias.

---

# 5. Quan hệ với RMSNorm

Một phát hiện quan trọng trong PaLM:

> RMSNorm và Feedforward Bias-Free hoạt động rất tốt khi kết hợp với nhau.

RMSNorm:

$$
RMS(x)= \sqrt{\frac1d \sum_i x_i^2}
$$

$$
y=\frac{x}{RMS(x)}
$$

không thực hiện centering như LayerNorm.

Do đó:

* kiến trúc đơn giản hơn
* ít phép toán hơn
* không cần bias bù trừ

Pipeline trở thành:

$$
x \rightarrow RMSNorm \rightarrow FFN(NoBias)
$$

---

# 6. Tác động tới tối ưu hóa

## 6.1 Gradient đơn giản hơn

FFN chuẩn:

$$
y=W_2\phi(W_1x+b_1)+b_2
$$

Gradient phải cập nhật:

$$
\nabla W_1
$$

$$
\nabla W_2
$$

$$
\nabla b_1
$$

$$
\nabla b_2
$$

Bias-free chỉ còn:

$$
\nabla W_1
$$

$$
\nabla W_2
$$

làm giảm lượng trạng thái optimizer.

---

## 6.2 Giảm Optimizer States

Với Adam:

$$
m_t
$$

$$
v_t
$$

được lưu cho mọi tham số.

Loại bỏ bias giúp giảm:

* tham số
* momentum
* variance estimates

tương ứng.

---

# 7. Tăng Throughput huấn luyện

Boris Dayma thực hiện nhiều thử nghiệm thực nghiệm cho thấy:

* tốc độ huấn luyện tăng
* throughput GPU tăng
* không suy giảm đáng kể độ chính xác

Nguyên nhân:

1. ít tham số hơn

2. ít truy cập bộ nhớ hơn

3. ít kernel operations hơn

4. giảm memory bandwidth pressure

Trong các LLM hiện đại:

> Memory bandwidth thường là nút thắt lớn hơn FLOPs.

Do đó giảm truy cập bộ nhớ thường đem lại lợi ích thực tế rõ rệt.

---

# 8. Liên hệ với PaLM

PaLM là một trong những hệ thống quy mô cực lớn đầu tiên phổ biến xu hướng:

* RMSNorm
* SwiGLU
* Bias-Free Feedforward

Thiết kế này cho thấy:

> Transformer không nhất thiết cần bias để đạt hiệu năng SOTA.

Triết lý thiết kế chuyển từ:

> Thêm nhiều thành phần để tăng biểu diễn

sang:

> Loại bỏ các thành phần không cần thiết để tăng hiệu quả mở rộng.

---

# 9. Tích hợp trong x-transformers

Trong x-transformers:

```python
Decoder(
    dim = 512,
    depth = 6,
    heads = 8,
    ff_no_bias = True
)
```

khi:

```python
ff_no_bias = True
```

mọi lớp Feedforward sẽ sử dụng:

$$
W_1x
$$

thay vì:

$$
W_1x+b_1
$$

và:

$$
W_2h
$$

thay vì:

$$
W_2h+b_2
$$

---

# 10. Phân tích hình học

Một neuron có bias:

$$
f(x)=w^Tx+b
$$

siêu phẳng quyết định:

$$
w^Tx+b=0
$$

Nếu bỏ bias:

$$
w^Tx=0
$$

siêu phẳng luôn đi qua gốc tọa độ.

Trong mạng nông điều này có thể hạn chế biểu diễn.

Tuy nhiên trong Transformer sâu:

* Residual Connections
* Normalization
* Attention Mixing
* Feedforward Stacking

đã tạo đủ độ linh hoạt.

Do đó mất bias không làm giảm đáng kể khả năng biểu diễn toàn mạng.

---

# 11. Vai trò trong xu hướng Minimal Transformer

Bias-Free FFN là một phần của phong trào:

## Minimal Transformer Design

bao gồm:

* RMSNorm
* Pre-Norm
* SwiGLU
* RoPE
* Bias-Free FFN
* Simplified Initialization

Mục tiêu:

$$
\text{Performance} \rightarrow \text{Maximum}
$$

$$
\text{Complexity} \rightarrow \text{Minimum}
$$

---

# 12. Ưu điểm và hạn chế

## Ưu điểm

* Ít tham số hơn
* Tăng throughput
* Giảm memory bandwidth
* Giảm optimizer states
* Dễ mở rộng mô hình lớn
* Hoạt động tốt với RMSNorm
* Được kiểm chứng trên PaLM và các LLM hiện đại

## Hạn chế

* Lợi ích nhỏ ở mô hình nhỏ
* Có thể giảm linh hoạt trong một số bài toán đặc thù
* Không phải lúc nào cũng cải thiện chất lượng
* Chủ yếu tối ưu hóa hiệu quả hệ thống

---

# 13. Kết luận

__No Bias Feedforward__ là một bước tiến trong quá trình tối giản hóa Transformer hiện đại. Các nghiên cứu gần đây cho thấy phần lớn năng lực biểu diễn của FFN đến từ ma trận trọng số và hàm kích hoạt, trong khi bias đóng góp rất ít sau các tầng chuẩn hóa như RMSNorm hoặc LayerNorm.

Việc loại bỏ bias giúp giảm số tham số, giảm chi phí bộ nhớ, tăng throughput huấn luyện và cải thiện khả năng mở rộng trên các hệ thống huấn luyện quy mô cực lớn. Đây là lý do nhiều kiến trúc hiện đại như PaLM và x-transformers đã xem __Bias-Free Feedforward __như một lựa chọn mặc định trong thiết kế Transformer thế hệ mới.

---

# Tài liệu tham khảo

1. Touvron et al., *No Train No Gain: Revisiting Efficient Training Algorithms for Transformer Models*, 2022.

2. Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022.

3. lucidrains, *x-transformers Repository*.

4. Boris Dayma, Experimental Notes on Transformer Architecture Simplification.

5. Vaswani et al., *Attention Is All You Need*, 2017.

6. Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.

# Thư viện tham khảo

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
        ff_no_bias = True  # set this to True
    )
)
```