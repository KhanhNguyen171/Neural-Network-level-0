# Transformer-XL Recurrence Mechanism

> **Segment-Level Recurrence for Long-Context Transformers**


## Mục lục

1. Giới thiệu
2. Động cơ nghiên cứu
3. Ý tưởng cốt lõi
4. Segment-Level Recurrence
5. Kiến trúc Attention với Memory
6. Relative Positional Encoding
7. Thuật toán huấn luyện
8. Suy diễn (Inference)
9. Độ phức tạp tính toán
10. Triển khai trong `x-transformers`
11. Ưu điểm và hạn chế
12. Kết luận
13. Tài liệu tham khảo

---

# 1. Giới thiệu

Mô hình Transformer chuẩn của Vaswani et al. (2017) chỉ có thể xử lý một ngữ cảnh hữu hạn:

$$
\mathbf{x}= (x_1,x_2,\dots,x_L)
$$

với:

* $L$: độ dài cửa sổ ngữ cảnh (context window).

Khi chuỗi đầu vào dài hơn $L$, các token trước đó bị loại bỏ hoàn toàn. Điều này tạo ra ba hạn chế cơ bản:

1. **Context Fragmentation**
2. **Long-Term Dependency Failure**
3. **Inefficient Computation on Long Sequences**

Để giải quyết vấn đề này, Dai et al. (2019) đề xuất **Transformer-XL**, giới thiệu cơ chế:

> **Segment-Level Recurrence Mechanism**

giúp mô hình duy trì thông tin vượt ra ngoài cửa sổ attention cố định.

---

# 2. Động cơ nghiên cứu

Giả sử chuỗi đầu vào:

```text
x1 x2 x3 x4 ... x10000
```

Transformer chuẩn phải chia thành các segment:

```text
Segment 1 : x1      ... x512
Segment 2 : x513    ... x1024
Segment 3 : x1025   ... x1536
```

Khi xử lý:

```text
Segment 2
```

mọi thông tin từ:

```text
Segment 1
```

đã bị mất hoàn toàn.

Do đó:

$$
P(x_t) \neq P(x_t|x_{<t})
$$

mà thực tế chỉ là:

$$
P(x_t|x_{t-L:t-1})
$$

gây suy giảm khả năng học phụ thuộc dài hạn.

---

# 3. Ý tưởng cốt lõi của Transformer-XL

Transformer-XL lưu lại hidden states của segment trước như một **memory bank**.

## Kiến trúc tổng quát

```text
Segment t-1
┌──────────────────┐
│ h1 h2 ... hL     │
└──────────────────┘
          │
          ▼
      Memory Bank
          │
          ▼
Segment t
┌──────────────────┐
│ x1 x2 ... xL     │
└──────────────────┘
```

Thay vì attention chỉ trên:

$$
\mathbf{H}_t
$$

mô hình attention trên:

$$
[\mathbf{M}_{t-1};\mathbf{H}_t]
$$

trong đó:

* $\mathbf{M}_{t-1}$: memory từ segment trước.
* $\mathbf{H}_t$: hidden states hiện tại.

---

# 4. Segment-Level Recurrence

## Memory Definition

Giả sử:

$$
\mathbf{H}_{t-1}= [h_1,h_2,\dots,h_L]
$$

Memory:

$$
\mathbf{M}_{t-1}= [h_{L-M+1},\dots,h_L]
$$

với:

* $M$: memory length.

Sau mỗi segment:

$$
\mathbf{M}_{t}= SG(\mathbf{H}_t)
$$

trong đó:

$$
SG(\cdot)
$$

là phép:

```python
detach()
```

để ngăn gradient lan truyền ngược qua các segment cũ.

---

# 5. Attention với Memory

## Transformer chuẩn

```text
x1 x2 x3 x4 x5
│  │  │  │  │
└──── attention ───┘
```

---

## Transformer-XL

```text
Memory:
m1 m2 m3 m4

Current:
x1 x2 x3 x4 x5
```

Token:

```text
x5
```

có thể attention tới:

```text
m1 m2 m3 m4 x1 x2 x3 x4
```

---

## Attention Formulation

Queries:

$$
Q=\mathbf{H}_tW_Q
$$

Keys:

$$
K=[\mathbf{M}_{t-1};\mathbf{H}_t]W_K
$$

Values:

$$
V=[\mathbf{M}_{t-1};\mathbf{H}_t]W_V
$$

Attention:

$$
\text{Attn}(Q,K,V)= \text{softmax} \left( \frac{QK^T}{\sqrt d} \right)V
$$

---

# 6. Relative Positional Encoding

Absolute positional embedding không thể hoạt động với recurrence.

Ví dụ:

```text
Segment 1:
position 1...512

Segment 2:
position 1...512
```

Memory sẽ gây xung đột vị trí.

Transformer-XL thay thế bằng:

## Relative Position Encoding

Điểm attention:

$$
A_{i,j}= q_i^T k_j + q_i^T r_{i-j} + u^T k_j + v^T r_{i-j}
$$

trong đó:

* $q_i^Tk_j$: content-content term.
* $q_i^Tr_{i-j}$: content-position term.
* $u^Tk_j$: global content bias.
* $v^Tr_{i-j}$: global positional bias.

Relative Position Encoding giúp memory có thể tái sử dụng giữa các segment mà không phá vỡ thông tin vị trí.

---

# 7. Thuật toán huấn luyện

## Forward

```text
Segment 1
      │
      ▼
 Memory 1
      │
      ▼
Segment 2
      │
      ▼
 Memory 2
      │
      ▼
Segment 3
```

---

## Pseudocode

```python
mem = None

for segment in segments:

    logits, mem = model(
        segment,
        mems=mem,
        return_mems=True
    )
```

---

## Gradient Flow

```text
Forward:
Past → Present

Backward:
Present only
```

Do:

```python
memory = memory.detach()
```

để tránh:

* exploding computational graph;
* GPU out-of-memory;
* gradient instability.

---

# 8. Effective Context Length

Transformer:

$$
L
$$

Transformer-XL:

$$
L_{effective}=L+M
$$

Ví dụ:

```python
max_seq_len = 512
max_mem_len = 2048
```

suy ra:

$$
L_{effective}= 2560
$$

tokens.

---

# 9. Độ phức tạp tính toán

Transformer chuẩn:

$$
O(L^2)
$$

Transformer-XL:

$$
O(L(L+M))
$$

trong đó:

* $L$: chiều dài segment.
* $M$: chiều dài memory.

Do:

$$
M \ll T
$$

với (T) là chiều dài toàn bộ chuỗi nên chi phí thấp hơn rất nhiều so với việc attention trên toàn bộ lịch sử.

---

# 10. Triển khai trong x-transformers

## Khởi tạo

```python
from x_transformers import (
    TransformerWrapper,
    Decoder
)

model_xl = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 512,
    max_mem_len = 2048,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rel_pos_bias = True
    )
)
```

---

## Forward qua nhiều segment

```python
logits1, mems1 = model_xl(
    seg1,
    return_mems = True
)

logits2, mems2 = model_xl(
    seg2,
    mems = mems1,
    return_mems = True
)

logits3, mems3 = model_xl(
    seg3,
    mems = mems2,
    return_mems = True
)
```

---

## XLAutoregressiveWrapper

```python
from x_transformers import (
    XLAutoregressiveWrapper
)

xl_wrapper = XLAutoregressiveWrapper(
    model_xl
)
```

Wrapper tự động:

1. chia segment;
2. quản lý memory;
3. huấn luyện trên chuỗi dài;
4. sinh văn bản dài vượt quá `max_seq_len`.

---

# 11. Minh họa tổng quát

```text
Long Sequence
──────────────────────────────────────

Segment1     Segment2     Segment3
┌───────┐    ┌───────┐    ┌───────┐
│Tokens │──► │Tokens │──► │Tokens │
└───────┘    └───────┘    └───────┘
      │            │            │
      ▼            ▼            ▼
   Memory1      Memory2      Memory3

Effective Context:

Memory + Current Segment
```

---

# 12. Ưu điểm

### Long-range Dependency

Có thể ghi nhớ hàng nghìn token.

### Efficient Inference

Không cần tính lại toàn bộ attention.

### Better Language Modeling

Giảm perplexity đáng kể.

### Streaming Generation

Phù hợp với:

* Large Language Models;
* Code Generation;
* Dialogue Systems;
* Speech Models.

---

# 13. Hạn chế

## Memory Cost

$$
O(M)
$$

---

## Gradient Truncation

Không thể lan truyền gradient vô hạn.

---

## Information Compression

Memory chỉ lưu hidden states, không lưu toàn bộ lịch sử đầu vào.

---

# 14. Vai trò trong sự phát triển của Long-Context Transformers

Transformer-XL là nền tảng cho:

* Memorizing Transformer
* RETRO
* RMT
* LongNet
* Infini-Attention
* Ring Attention
* External Memory Transformer

Đóng góp quan trọng nhất:

> Tách biệt giữa **attention window** và **effective context length**, mở đường cho các kiến trúc Transformer ngữ cảnh dài hiện đại.

---

# Tài liệu tham khảo

```bibtex
@article{dai2019transformerxl,
  title={Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc and Salakhutdinov, Ruslan},
  journal={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={2978--2988},
  year={2019}
}
```

```bibtex
@misc{wang2023xtransformers,
  title={x-transformers},
  author={Phil Wang},
  howpublished={\url{https://github.com/lucidrains/x-transformers}},
  year={2023}
}
```

```bibtex
@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and others},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```
