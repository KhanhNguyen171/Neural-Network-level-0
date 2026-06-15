# image transfoermers & Distilation attention

<img src="assets/distill.png">

Điểm mới không nằm ở Kiến trúc Transformer mà nằm ở __Distilation Token Và Attention-based Distilation Mechanism__

# Data-efficient Image Transformers (DeiT)

$$\text{ViT + Data-Effcient Training + Knowledge Distilation}$$

# 1. Giới thiệu

DeiT (Data-efficient Image Transformer) được đề xuất nhằm giải quyết một hạn chế quan trọng của Vision Transformer:

> ViT thường yêu cầu tập dữ liệu cực lớn (JFT-300M, ImageNet-21K) để đạt hiệu năng cao.

Bài báo chứng minh rằng Vision Transformer có thể được huấn luyện trực tiếp trên ImageNet-1K mà vẫn đạt hiệu năng cạnh tranh với CNN hiện đại thông qua:

* Training Recipe tối ưu
* Knowledge Distillation
* Distillation Attention

Không cần dữ liệu ngoài ImageNet-1K.

---

# 2. Kiến trúc tổng thể

<img src="assets/deit_architecture.png">

Kiến trúc DeiT kế thừa hoàn toàn Vision Transformer.

Input image:

$$
x \in \mathbb{R}^{H \times W \times C}
$$

được chia thành các patch:

$$
x \rightarrow {x_1,x_2,\ldots,x_N}
$$

Mỗi patch được ánh xạ tuyến tính:

$$
z_i = E x_i
$$

với:

$$
E \in \mathbb{R}^{D \times (P^2C)}
$$

---

## Token Sequence

Thay vì chỉ sử dụng CLS token như ViT:

$$
\text{CLS},z_1,z_2,\ldots,z_N
$$

DeiT bổ sung thêm một token mới:

$$
\text{CLS},\text{DIST},z_1,z_2,\ldots,z_N
$$

Trong đó:

* CLS token học thông tin phục vụ classification
* DIST token học thông tin từ Teacher Model

---

# 3. Patch Embedding

Số lượng patch:

$$
N = \frac{HW}{P^2}
$$

Embedding đầu vào:

$$
h_i^{(0)}= z_i+p_i
$$

với:

* $z_i$: patch embedding
* $p_i$: positional embedding

---

# 4. Transformer Encoder

Kiến trúc encoder hoàn toàn giống ViT.

Mỗi block gồm:

## Multi-Head Self-Attention

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Attention:

$$
\text{Attention}(Q,K,V) =\text{Softmax} \left( \frac{QK^T}{\sqrt d} \right)V
$$

Toàn bộ token:

$$
\text{CLS},\text{DIST},z_1,\ldots,z_N
$$

đều tham gia attention.

Do đó Distillation Token có thể tương tác trực tiếp với:

* CLS Token
* Patch Tokens

thông qua self-attention.

---

# 5. Output Heads

Sau L Transformer layers:

$$
h^{(L)} = h_{cls},h_{dist},h_1,\ldots,h_N
$$

DeiT sử dụng hai classifier riêng biệt.

---

## Classification Head

CLS Token:

$$
y_{cls}= W_{cls}h_{cls}
$$

---

## Distillation Head

Distillation Token:

$$
y_{dist} = W_{dist}h_{dist}
$$

---

# 6. Distillation Attention

Đây là đóng góp quan trọng nhất của bài báo.

---

## Knowledge Distillation truyền thống

Teacher model:

$$
f_T(x)
$$

Student model:

$$
f_S(x)
$$

Student học:

$$
L_{KD} = KL \left( f_T(x), f_S(x) \right)
$$

Distillation chỉ xảy ra tại output logits.

---

## Distillation trong DeiT

Thay vì ép toàn bộ mạng học từ teacher output trực tiếp.

Tác giả thêm Distillation Token:

$$
t_{dist}
$$

Token này được truyền qua toàn bộ Transformer.

Do đó teacher information được lan truyền thông qua:

$$
\text{Self-Attention}
$$

thay vì chỉ tại tầng cuối.

---

## Attention-based Distillation

Distillation Token hoạt động như một truy vấn học được:

$$
q_{dist}
$$

tương tác với mọi patch:

$$
A_{dist}= \text{Softmax} \left( \frac {q_{dist}K^T} {\sqrt d} \right)
$$

Thông tin teacher được tích hợp vào biểu diễn:

$$
h_{dist}
$$

trong suốt quá trình attention.

Đây là lý do tác giả gọi phương pháp này là:

> Distillation through Attention

---

# 7. Loss Function

Loss tổng:

$$
L=L_{cls} + L_{dist}
$$

---

## Classification Loss

$$
L_{cls} =CE(y_{cls},y)
$$

---

## Distillation Loss

$$
L_{dist} =CE(y_{dist},y_T)
$$

với:

$$
y_T =f_T(x)
$$

---

## Tổng Loss

$$
L=CE(y_{cls},y)+CE(y_{dist},y_T)
$$

---

# 8. Hard Distillation

Bài báo phát hiện một kết quả bất ngờ.

Teacher prediction dạng one-hot:

$$
\hat y_T =\arg\max f_T(x)
$$

lại hiệu quả hơn soft-label distillation.

Khi đó:

$$
L_{dist} =CE(y_{dist},\hat y_T)
$$

được gọi là:

> Hard Distillation

Đây là thiết lập đạt kết quả tốt nhất trong DeiT.

---

# 9. Soft Distillation

Teacher logits:

$$
z_T
$$

Student logits:

$$
z_S
$$

Xác suất mềm:

$$
p_T =\text{Softmax}\left(\frac{z_T}{\tau}\right)
$$

$$
p_S =\text{Softmax}\left(\frac{z_S}{\tau}\right)
$$

Loss:

$$
L_{KD} =\tau^2KL(p_T||p_S)
$$

Trong DeiT, soft distillation không vượt qua hard distillation.

---

# 10. Inference

Trong quá trình suy luận:

CLS Head:

$$
y_{cls}
$$

Distillation Head:

$$
y_{dist}
$$

được trung bình:

$$
y=\frac {y_{cls}+y_{dist}} {2}
$$

---

# 11. Ý nghĩa toán học

Distillation Token tạo thêm một vector biểu diễn:

$$
h_{dist} \in \mathbb{R}^D
$$

học:

$$
h_{dist} =f(x,\text{Teacher Knowledge})
$$

trong khi:

$$
h_{cls} = f(x)
$$

Do attention kết nối toàn cục:

$$
h_{dist} \leftrightarrow h_i
$$

với mọi patch token.

Teacher information được lan truyền xuyên suốt toàn bộ Transformer thay vì chỉ tác động tại output layer.

---

# 12. Đóng góp khoa học

1. Chứng minh ViT có thể huấn luyện hiệu quả trên ImageNet-1K.
2. Giới thiệu Distillation Token.
3. Đề xuất Distillation through Attention.
4. Hard Distillation hiệu quả hơn Soft Distillation.
5. Đạt hiệu năng cạnh tranh với CNN hiện đại mà không cần dữ liệu khổng lồ.

---

# 13. Tóm tắt

$$
\boxed{\text{DeiT} = \text{ViT} + \text{Distillation Token} + \text{Attention-based Distillation} }
$$

Điểm cốt lõi của DeiT không phải là thay đổi kiến trúc Transformer, mà là đưa kiến thức từ Teacher Model vào trong quá trình Self-Attention thông qua một Distillation Token chuyên biệt.


---

**Tài liệu tham khảo**

```bibtex
@misc{touvron2020training,
    title   = {Training data-efficient image transformers & distillation through attention},
    author  = {Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
    year    = {2020},
    eprint  = {2012.12877},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
    
    Link: $$https://arxiv.org/pdf/2012.12877$$
}
```

---

**Thư viện tham khảo**

```python
import torch
from torchvision.models import resnet50

from vit_pytorch.distill import DistillableViT, DistillWrapper

teacher = resnet50(pretrained = True)

v = DistillableViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)

img = torch.randn(2, 3, 256, 256)
labels = torch.randint(0, 1000, (2,))

loss = distiller(img, labels)
loss.backward()

# after lots of training above ...

pred = v(img) # (2, 1000)
```

Lớp `DistillableViT` giống y hệt `ViT` trừ cách xử lý forward pass, nên bạn có thể load lại trọng số (parameters) về `ViT` sau khi train distillation xong.

Ngoài ra, chỉ cần dùng hàm `.to_vit` trên đối tượng `DistillableViT` là có thể lấy lại ngay một đối tượng `ViT`.

```python
v = v.to_vit()
type(v) # <class 'vit_pytorch.vit_pytorch.ViT'>
```