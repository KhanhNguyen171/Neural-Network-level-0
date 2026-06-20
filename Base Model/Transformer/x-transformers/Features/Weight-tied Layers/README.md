# Weight-Tied Layers in Transformers

> Parameter Sharing Across Depth for Efficient Deep Transformers

---

# 1. Giới thiệu

Trong Transformer nguyên bản [1], mỗi tầng (layer) sở hữu một bộ tham số riêng biệt:

$$
h_{l+1} = F_l(h_l)
$$

trong đó:

* $h_l$ là biểu diễn tại tầng $l$
* $F_l$ là Transformer Block thứ $l$

Điều này dẫn tới số lượng tham số tăng tuyến tính theo độ sâu mạng:

$$
\text{Params} = O(LP)
$$

với:

* $L$ là số tầng
* $P$ là số tham số của mỗi tầng

Khi Transformer trở nên rất sâu (24–100+ layers), phần lớn tham số được quan sát là dư thừa và học các phép biến đổi tương tự nhau.

Để giải quyết vấn đề này, ALBERT [^2] đề xuất:

> Chia sẻ cùng một bộ tham số cho nhiều tầng Transformer.

Ý tưởng này được gọi là **Weight-Tied Layers** hay **Cross-Layer Parameter Sharing**.

---

# 2. Nguyên lý cốt lõi

## Transformer thông thường

Mỗi tầng có tham số riêng:

$$
h_1 = F(h_0;\theta_1)
$$

$$
h_2 = F(h_1;\theta_2)
$$

$$
h_3 = F(h_2;\theta_3)
$$

$$
\vdots
$$

$$
h_L = F(h_{L-1};\theta_L)
$$

Trong đó:

$$
\theta_1 \neq \theta_2 \neq \cdots \neq \theta_L
$$

---

## Weight-Tied Transformer

Tất cả tầng sử dụng cùng một tập trọng số:

$$
\theta_1=\theta_2=\cdots=\theta_L=\theta
$$

Do đó:

$$
h_{l+1}=F(h_l;\theta)
$$

với mọi:

$$
l=0,\ldots,L-1
$$

Toàn bộ Transformer trở thành:

$$
h_L = F^L(h_0)
$$

hay:

$$
h_L = \underbrace{ F(F(F(\cdots F}_{L\ \text{lần}} (h_0))))
$$

---

# 3. Trực giác hình học

Transformer thông thường:

```text
Input
 │
 ▼
Layer 1 (θ₁)
 │
 ▼
Layer 2 (θ₂)
 │
 ▼
Layer 3 (θ₃)
 │
 ▼
Output
```

---

Weight-Tied Transformer:

```text
Input
 │
 ▼
Shared Layer (θ)
 │
 ▼
Shared Layer (θ)
 │
 ▼
Shared Layer (θ)
 │
 ▼
Output
```

Từ góc nhìn động lực học:

$$
x_{t+1}=F_\theta(x_t)
$$

Transformer không còn là một chuỗi các hàm khác nhau mà trở thành một quá trình **lặp đi lặp lại cùng một phép biến đổi**.

---

# 4. Liên hệ với Recurrent Neural Networks

Weight Tying tạo ra sự tương đồng mạnh với RNN.

## RNN

$$
h_t = f(h_{t-1},x_t;\theta)
$$

Cùng một tham số được sử dụng theo thời gian.

---

## Weight-Tied Transformer

$$
h_{l+1}=F(h_l;\theta)
$$

Cùng một tham số được sử dụng theo chiều sâu.

Do đó có thể xem Weight-Tied Transformer là:

> Recurrent Neural Network trong không gian độ sâu (depth dimension).

---

# 5. Các thành phần được chia sẻ

Thông thường toàn bộ Transformer Block được chia sẻ.

## Self-Attention

Projection:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Các ma trận:

$$
W_Q,;W_K,;W_V
$$

được tái sử dụng ở mọi tầng.

---

## Multi-Head Attention

$$
\text{Attention}(Q,K,V)= \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Mọi head đều sử dụng cùng tập trọng số xuyên suốt mạng.

---

## Feed Forward Network

$$
\text{FFN}(x)= W_2 \sigma(W_1x)
$$

với:

$$
W_1,W_2
$$

được chia sẻ giữa các tầng.

---

## LayerNorm

Tùy thiết kế:

* Có thể chia sẻ
* Có thể độc lập

ALBERT chủ yếu chia sẻ toàn bộ Transformer Block.

---

# 6. Phân tích động lực học

Xét phép biến đổi:

$$
h_{t+1}=F(h_t;\theta)
$$

Sau $L$ bước:

$$
h_L=F^L(h_0)
$$

Nếu tồn tại điểm cố định:

$$
h^*=F(h^*)
$$

thì:

$$
\lim_{L\to\infty} F^L(h_0)= h^*
$$

Điều này cho thấy Weight-Tied Layers có mối liên hệ trực tiếp với:

* Fixed Point Iteration
* Deep Equilibrium Models
* Universal Transformers

---

# 7. Gradient của Weight-Tied Layers

Trong Transformer thông thường:

$$
\frac{\partial \mathcal L} {\partial \theta_l}= \frac{\partial \mathcal L} {\partial h_l} \frac{\partial h_l} {\partial \theta_l}
$$

Mỗi tầng cập nhật độc lập.

---

Trong Weight Tying:

$$
\frac{\partial \mathcal L} {\partial \theta}= \sum_{l=1}^{L} \frac{\partial \mathcal L} {\partial h_l} \frac{\partial h_l} {\partial \theta}
$$

Một tham số nhận tín hiệu gradient từ tất cả các tầng.

Điều này tạo nên:

* Regularization tự nhiên
* Khả năng tổng quát hóa tốt hơn
* Giảm overfitting

---

# 8. Layer Recurrence

Một mở rộng quan trọng của Weight Tying là Layer Recurrence.

Thay vì:

```text
A
B
C
```

ta có:

```text
A → A → A → A
↓
B → B → B → B
↓
C → C → C → C
```

Mỗi block được lặp lại nhiều lần trước khi chuyển sang block tiếp theo.

Ý tưởng này xuất hiện trong nhiều nghiên cứu gần đây về:

* Recurrent Transformers
* Universal Transformers
* Test-Time Scaling

---

# 9. Triển khai trong x-transformers

Chia sẻ toàn bộ tầng:

```python
Encoder(
    dim = 512,
    depth = 12,
    weight_tie_layers = True
)
```

Tương đương:

```text
θ
↺
12 lần
```

---

Layer Recurrence:

```python
layers_execute_order = (
    *((0, 1) * 4),
    *((2, 3) * 4),
    *((4, 5) * 4),
)
```

Sơ đồ thực thi:

```text
(A,F)
(A,F)
(A,F)
(A,F)

(B,F)
(B,F)
(B,F)
(B,F)

(C,F)
(C,F)
(C,F)
(C,F)
```

Trong đó:

* $A$ = Attention
* $F$ = FeedForward

---

# 10. Iterative Refinement Hypothesis

ALBERT đưa ra một quan sát quan trọng:

> Transformer không nhất thiết cần nhiều phép biến đổi khác nhau; một phép biến đổi tốt có thể được áp dụng nhiều lần để liên tục tinh chỉnh biểu diễn.

Quá trình:

```text
Pass 1
 ↓
Pass 2
 ↓
Pass 3
 ↓
Pass 4
```

có thể được xem như:

$$
h^{(t+1)}= F(h^{(t)})
$$

Biểu diễn được cải thiện dần sau mỗi lần lặp.

---

# 11. Liên hệ với Universal Transformer

Universal Transformer [3] mở rộng ý tưởng này bằng cách:

$$
h_{t+1}= F(h_t) + \text{Position}(t)
$$

Trong đó:

* Một block duy nhất được lặp lại nhiều lần
* Mô hình học số bước suy luận động

Weight-Tied Layers có thể xem là tiền đề trực tiếp của Universal Transformer.

---

# 12. Liên hệ với Deep Equilibrium Models

Deep Equilibrium Models (DEQ) [4] thay thế:

$$
F^L(x)
$$

bằng việc tìm nghiệm:

$$
h^*=F(h^*)
$$

Thông qua thuật toán fixed-point.

Lộ trình phát triển:

```text
Transformer
     │
     ▼
ALBERT
     │
     ▼
Weight Tying
     │
     ▼
Universal Transformer
     │
     ▼
Layer Recurrence
     │
     ▼
Deep Equilibrium Models
```

---

# 13. Ưu điểm

## Giảm số tham số

Từ:

$$
O(LP)
$$

thành:

$$
O(P)
$$

---

## Tăng hiệu quả bộ nhớ

Có thể xây dựng Transformer rất sâu mà không tăng đáng kể số lượng tham số.

---

## Regularization tự nhiên

Các tầng không thể ghi nhớ chức năng riêng biệt.

Mô hình buộc phải học:

$$
F_\theta
$$

mang tính tổng quát hơn.

---

## Hỗ trợ suy luận lặp

Cho phép:

* Iterative Refinement
* Test-Time Scaling
* Recurrent Reasoning

---

# 14. Hạn chế

## Giảm khả năng chuyên môn hóa

Transformer thông thường có thể hình thành:

* tầng thấp → cú pháp
* tầng giữa → ngữ nghĩa
* tầng cao → suy luận

Weight Tying làm giảm sự phân hóa này.

---

## Nguy cơ hội tụ sớm

Nếu:

$$
F(F(x)) \approx F(x)
$$

thì nhiều lần lặp có thể không mang lại thêm thông tin.

---

## Biểu diễn kém đa dạng hơn

Mọi tầng đều bị ràng buộc bởi cùng một hàm biến đổi.

---

# 15. Kết luận

Weight-Tied Layers là một trong những ý tưởng quan trọng nhất trong hướng nghiên cứu **parameter-efficient Transformers**.

Thay vì xem Transformer là chuỗi các tầng độc lập:

$$
F_1,F_2,\ldots,F_L
$$

Weight Tying xem Transformer như một quá trình lặp:

$$
h_{t+1}=F(h_t;\theta)
$$

với cùng một bộ tham số được áp dụng nhiều lần.

Tư tưởng này tạo nền tảng cho:

* ALBERT
* Universal Transformer
* Recurrent Transformer
* Layer Recurrence
* Deep Equilibrium Models
* x-transformers

và là một trong những hướng tiếp cận quan trọng để xây dựng các Transformer sâu hơn, hiệu quả hơn và có khả năng suy luận lặp trong các hệ thống thế hệ mới.

---

# Tài liệu tham khảo

[1]: Vaswani, A., et al. *Attention Is All You Need*. NeurIPS 2017.

[2]: Lan, Z., et al. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. 2019. https://arxiv.org/abs/1909.11942

[3]: Dehghani, M., et al. *Universal Transformers*. ICLR 2019.

[4]: Bai, S., Kolter, J. Z., Koltun, V. *Deep Equilibrium Models*. NeurIPS 2019.

[5]: x-transformers implementation. https://github.com/lucidrains/x-transformers

[6]: Anonymous. *The Unreasonable Ineffectiveness of the Deeper Layers*. 2024. https://arxiv.org/pdf/2405.15071
