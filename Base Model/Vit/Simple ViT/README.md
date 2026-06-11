# Cải tiến đơn giản hóa cho ViT base

chỉ với một số thay đổi nhỏ trong pipeline huấn luyện, một __Vision Transformer__ thuần túy có thể đạt hiệu năng cạnh tranh trên __ImageNet-1K__ mà không cần các kỹ thuật __regularization__ phức tạp.


- Sử dụng nhúng vị trí hình sin 2D (__2d sinusoidal positional embedding__).

- Sử dụng pooling trung bình toàn cục (__global average pooling__) và loại bỏ mã thông báo CLS (no CLS token).

- Không sử dụng dropout.

- Giảm kích thước batch (__batch size__) xuống 1024 thay vì 4096.

- Sử dụng các phương pháp tăng cường dữ liệu __RandAugment__ và __MixUp__.

---

# Vision Transformer Gốc

## 1. Patch Embedding
Cho ảnh đầu vào:
$$X \in \mathbb{R}^{H \times W \times C}$$

Chia thành các patch kích thước:
$$P \times P$$

Số lượng patch:
$$N = \frac {HW} {P^2}$$

Mỗi patch được flatten:
$$x_i \in \mathbb{R}^{P^2 C}$$

Sau đó chiếu tuyến tính:
$$z_i = E x_i$$

Với:
$$E \in \mathbb{R}^{D \times P^2 C}$$

Thu được token embedding:
$$z_i \in \mathbb{R}^D$$

---

## 2. Positional Encoding
ViT gốc sử dụng positional embedding học được:
$$p_i$$

Token đầu vào:
$$h_j^{(0)} = z_i + p_i$$

---

## 3. Self-Attention
Mỗi block Transformer:
$$Q = XW_Q$$
$$K = XW_K$$
$$V = XW_V$$

Attention:
$$Attention(Q, K, V) = Softmax(\frac {QK^T} {\sqrt{d}}V)$$

Độ phức tạp:
$$O(N^2)$$

Với:
$$N = \text{số patch}$$

---

# Vấn đề của Vit Trên ImageNet-1K

ImagetNet-1K:
$$1.28M$$
ảnh huấn luyện.

Kích thước này nhỏ hơn nhiều so với:
- JFT-300M
- ImagetNet-21k

Do đó ViT thường:
- overfit
- học chậm
- hội tụ kém

nếu áp dụng recipe gốc.

---

# Triết lý của bài báo Better plain Vit Baselines for ImagerNet-1k
Thay vì thay đổi kiến trúc:
$$f_\theta (x)$$

Tác giả giữ nguyên:
- Multi-head Attention
- MLP Block
- Residual Connection

Và chỉ tối ưu:
$$\mathcal{T}$$

(Training Procedure)

Ta có:
$$Performance = f(Model, Data, Training)$$

Bài báo tập trung vào:
$$Training$$

## 1. Thay đổi Global Average Pooling
### Vit gốc
sử dụng token đặt biệt: $CLS$

Output: $y = h^{(L)}_{CLS}$, để thực hiện classification.

### Better Plain ViT
Tác giả loại  bỏ CLS Token.

sử dụn toàn bộ feature space:
$$z \in \mathbb{R}^D$$

Thay bằng:
$$z = \frac {1} {N} \sum^N_{i=1} h_i^{(L)}$$

Tính: $y = Wz$

Giảm phương sai estimator:
$$Var(\bar{X}) = \frac {\sigma^2} {N}$$

## 2. Thay đổi Fixed Sin-Cos Positional Encoding

Thay vì học: $p_i$

Tác giả dùng positional encoding cố định.

Cho vị trí: $Pos$, và chiều: $2k$

Ta có:
$$PE(pos, 2k) = \sin(\frac {pos} {10000^{2k/D}})$$

$$PE(pos, 2k+1) = \cos(\frac {pos} {10000^{2k/D}})$$

Không phải học thêm: $N \times D$ tham số, giảm overfitting và giữ được thông tin khoảng cách tương đối giữa các patch.

## 3. Thay đổi Batch Size nhỏ hơn
Gradient mini-batch:
$$g_B = \frac {1} {B} \sum^B_{i=1} \nabla_\theta L_i$$

Variance:
$$Var(g_B) = \frac {\sigma^2} {B}$$

Batch quá lớn:
$$Var(g_B) \rightarrow 0$$

Gradient trở nên quá "mượt".

## 4. Thay đổi Standard Augmentation
Sử dụng các kỹ thuật:
- Random Crop
- Random Flip
- RandAugment
- Mixup

Mixup:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

Với:
$$\lambda \sim Beta(\alpha, \alpha)$$

## 5. Hàm mất mát
Cross Entropy:
$$L = - \sum^C_{k=1} y_k \log p_k$$

Với:
$$p_k = \frac {e^{z_k}} {\sum_j e^{z_j}}$$

Gradient:
$$\frac {\partial L} {\partial z_k} = p_k - y_k$$

## 6. AdamW
Optimizer:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2_t$$

Bias Correction:
$$\hat{m}_t = \frac {m_t} {1 - \beta_1^t}$$
$$\hat{v}_t = \frac {v_t} {1 - \beta_2^t}$$

Update:
$$\theta_{t+1} = \theta_t - \eta \frac {\hat{m}_t} {\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

AdamW tách weight decay khỏi gradient giúp tối ưu hóa ổn định hơn cho Transformer.

---

**Tài liệu tham khảo**

```bibtex
@misc{Beyer2022BetterPlainViT
    title     = {Better plain ViT baselines for ImageNet-1k},
    author    = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
    publisher = {arXiv},
    year      = {2022},
    Link: [https://arxiv.org/pdf/2205.01580]
}

```

---

**Thư viện tham khảo**

```python
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```