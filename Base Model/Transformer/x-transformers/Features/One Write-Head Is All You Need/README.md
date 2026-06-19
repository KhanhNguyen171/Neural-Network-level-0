# One Write-Head Is All You Need

## Multi-Query Attention (MQA) và Grouped Query Attention (GQA)

> Nguồn chính:
>
> * Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019)
> * Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)

---

# 1. Giới thiệu

Một trong những nút thắt hiệu năng lớn nhất của Transformer hiện đại không nằm ở phép tính Attention mà nằm ở **KV Cache** trong quá trình suy luận tự hồi quy (*autoregressive decoding*).

Trong kiến trúc Multi-Head Attention (MHA) tiêu chuẩn, mỗi attention head sở hữu tập Key và Value riêng biệt.

$$
Q_i = XW_i^Q
$$

$$
K_i = XW_i^K
$$

$$
V_i = XW_i^V
$$

với

$$
i = 1,\dots,H
$$

trong đó:

* $H$: số attention heads
* $d_h$: kích thước mỗi head
* $X$: đầu vào của attention

Khi chiều dài ngữ cảnh tăng lên, toàn bộ các tensor Key và Value phải được lưu lại trong KV Cache.

Chi phí bộ nhớ tỷ lệ với:

$$
O(HN)
$$

trong đó:

* $H$ là số head
* $N$ là chiều dài ngữ cảnh

Đây chính là động lực dẫn tới sự ra đời của **Multi-Query Attention (MQA)**.

---

# 2. Multi-Head Attention

## Kiến trúc chuẩn

```text
                 Input X
                     |
      ---------------------------------
      |       |       |       |       |
    Head1   Head2   Head3   ...    HeadH
      |       |       |              |
     QKV     QKV     QKV           QKV
      |       |       |              |
      --------------------------------
                     |
                 Concat
                     |
                  Output
```

Mỗi head thực hiện:

$$
A_i= softmax \left( \frac{Q_iK_i^T} {\sqrt{d_h}} \right)
$$

$$
O_i=A_iV_i
$$

Sau đó:

$$
O= Concat (O_1,\dots,O_H)
$$

---

# 3. Vấn đề của KV Cache

Trong autoregressive decoding:

```text
Token 1 -> lưu K,V
Token 2 -> lưu K,V
Token 3 -> lưu K,V
...
Token N -> lưu K,V
```

Với MHA:

```text
Head1 -> K1,V1
Head2 -> K2,V2
Head3 -> K3,V3
...
HeadH -> KH,VH
```

Kích thước KV Cache:

$$
\text{Cache}= 2HN d_h
$$

(hệ số 2 tương ứng với Key và Value)

Khi:

$$
H=64
$$

và

$$
N=32768
$$

KV Cache trở thành thành phần tiêu thụ bộ nhớ lớn nhất trong hệ thống.

---

# 4. Multi-Query Attention (MQA)

## Ý tưởng cốt lõi

Shazeer (2019) đưa ra quan sát:

> Các Query Head cần đa dạng, nhưng Key và Value giữa các head thường mang thông tin tương đối giống nhau.

Do đó:

* Giữ nhiều Query Heads
* Chỉ sử dụng một Key Head
* Chỉ sử dụng một Value Head

---

## Kiến trúc MQA

```text
                     Input
                        |
       ----------------------------------
       |      |      |      |      |
      Q1     Q2     Q3    ...     QH
       |      |      |             |
       --------------------------------
                        |
                 Shared K,V
                        |
                  ------------
                  |          |
                  K          V
```

---

## Projection

Queries:

$$
Q_i = XW_i^Q
$$

với

$$
i=1,\dots,H
$$

Nhưng chỉ có:

$$
K=XW^K
$$

$$
V=XW^V
$$

---

## Attention

Head thứ $i$:

$$
A_i= softmax \left( \frac{Q_iK^T} {\sqrt{d_h}} \right)
$$

$$
O_i=A_iV
$$

Toàn bộ head sử dụng cùng một cặp Key/Value.

---

# 5. One Write-Head

Tên bài báo xuất phát từ quá trình ghi KV Cache.

Trong MHA:

```text
Head1 ghi K1,V1
Head2 ghi K2,V2
Head3 ghi K3,V3
...
HeadH ghi KH,VH
```

Có tổng cộng:

$$
H
$$

lần ghi.

Trong MQA:

```text
Shared K
Shared V
```

chỉ cần:

$$
1
$$

lần ghi.

Do đó:

```text
Many Query Heads
One Key Head
One Value Head
```

hay còn gọi là:

> One Write-Head.

---

# 6. Phân tích độ phức tạp

## Multi-Head Attention

KV Cache:

$$
O(HN)
$$

---

## Multi-Query Attention

KV Cache:

$$
O(N)
$$

---

## Memory Reduction

Giả sử:

$$
H=32
$$

Khi đó:

$$
\frac{O(N)} {O(HN)}= \frac{1}{32}
$$

Tức bộ nhớ giảm xấp xỉ:

$$
32\times
$$

---

# 7. Hạn chế của MQA

MQA đạt hiệu quả bộ nhớ rất cao nhưng tồn tại hạn chế:

```text
Tất cả Query Heads
phải chia sẻ cùng K,V
```

Do đó:

* Giảm tính đa dạng biểu diễn
* Giảm năng lực mô hình
* Có thể làm giảm chất lượng huấn luyện

Quan hệ tổng quát:

$$
\text{Quality(MHA)} > \text{Quality(MQA)}
$$

---

# 8. Grouped Query Attention (GQA)

## Động cơ

MHA:

```text
32 Query Heads
32 KV Heads
```

MQA:

```text
32 Query Heads
1 KV Head
```

GQA là điểm trung gian giữa hai cực này.

---

## Ý tưởng

Ví dụ:

```text
32 Query Heads
8 KV Heads
```

Mỗi KV Head được chia sẻ bởi nhiều Query Heads.

---

## Minh họa

```text
Q1
Q2
Q3
Q4
  \
   KV1

Q5
Q6
Q7
Q8
  \
   KV2

...

Q29
Q30
Q31
Q32
   \
    KV8
```

---

## Kiến trúc tổng quát

```text
                     Input
                        |
      -------------------------------------
      |      |      |      |      |      |
      Q1     Q2     Q3     Q4    ...   Q32
       \      |      |      /
              KV1

       \      |      |      /
              KV2

                  ...

              KV8
```

---

# 9. Công thức GQA

Giả sử:

$$
H_q= 32
$$

query heads

và

$$
H_{kv}= 8
$$

KV heads.

Tỷ lệ nhóm:

$$
g = \frac{H_q} {H_{kv}}
$$

Trong ví dụ:

$$
g=4
$$

Mỗi KV head phục vụ:

$$
4
$$

Query Heads.

---

Attention của head thứ $i$:

$$
A_i= Softmax \left( \frac{ Q_iK_{g(i)}^T } {\sqrt{d_h}} \right)
$$

$$
O_i=A_iV_{g(i)}
$$

trong đó:

$$
g(i)
$$

là KV group tương ứng của head $i$.

---

# 10. So sánh MHA, GQA và MQA

| Kiến trúc | Query Heads | KV Heads |
| --------- | ----------- | -------- |
| MHA       | $H$         | $H$      |
| GQA       | $H$         | $H_{kv}$ |
| MQA       | $H$         | $1$      |

---

## Bộ nhớ

$$
\text{MHA} > \text{GQA} > \text{MQA}
$$

---

## Khả năng biểu diễn

$$
\text{MHA} > \text{GQA} > \text{MQA}
$$

---

## Tốc độ suy luận

$$
\text{MQA} > \text{GQA} > \text{MHA}
$$

---

# 11. Vai trò trong x-transformers

MQA:

```python
Decoder(
    dim = 512,
    depth = 6,
    heads = 8,
    attn_one_kv_head = True
)
```

Tương đương:

$$
H_{kv}=1
$$

---

GQA:

```python
Decoder(
    dim = 512,
    depth = 12,
    heads = 8,
    attn_kv_heads = 2
)
```

Tương đương:

$$
H_q=8
$$

$$
H_{kv}=2
$$

Mỗi KV Head phục vụ:

$$
4
$$

Query Heads.

---

# 12. Từ MHA đến GQA

```text
                 Multi-Head Attention
                          |
                          |
                   KV Cache lớn
                          |
                          v
                Multi-Query Attention
                  (1 KV Head)
                          |
                          |
                Memory cực thấp
                          |
                          v
              Grouped Query Attention
                  (Few KV Heads)
                          |
                          |
               Cân bằng hiệu năng
               và chất lượng mô hình
                          |
                          v
           LLaMA-2 / Gemma / Mistral
                    / Gemini
```

---

# 13. Kết luận

Multi-Query Attention là một trong những cải tiến quan trọng nhất đối với Transformer hiện đại. Ý tưởng trung tâm là:

> Đa dạng hóa Query Heads nhưng chia sẻ Key và Value.

Điều này giúp giảm đáng kể chi phí KV Cache:

$$
O(HN) \rightarrow O(N)
$$

Từ MQA, cộng đồng nghiên cứu tiếp tục phát triển **Grouped Query Attention (GQA)** nhằm cân bằng giữa:

* hiệu quả bộ nhớ,
* tốc độ suy luận,
* năng lực biểu diễn.

Ngày nay GQA đã trở thành kiến trúc attention mặc định của nhiều Large Language Models hiện đại như LLaMA-2, Mistral, Gemma và Gemini.

# 14. References

## Papers

### [1] Fast Transformer Decoding: One Write-Head is All You Need

Noam Shazeer.

**arXiv:** 1911.02150, 2019.

```bibtex
@article{shazeer2019fast,
  title={Fast Transformer Decoding: One Write-Head is All You Need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
}
```

Link:

https://arxiv.org/abs/1911.02150

---

### [2] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai.

**EMNLP 2023**

```bibtex
@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebron, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023}
}
```

Link:

https://arxiv.org/abs/2305.13245

---

### [3] Multi-Query Attention with Shared Key and Value Projections

Ainslie et al.

Nghiên cứu này mở rộng ý tưởng MQA thành nhiều nhóm KV Heads, tạo tiền đề cho GQA được sử dụng trong các LLM hiện đại.

Link:

https://arxiv.org/pdf/2305.13245

---

### [4] Transformer Architecture

Ashish Vaswani et al.

**Attention Is All You Need**, NeurIPS 2017.

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

Link:

https://arxiv.org/abs/1706.03762

---

## x-transformers Implementation

### [5] x-transformers

Phil Wang (lucidrains)

Một triển khai thực nghiệm của nhiều biến thể Transformer hiện đại, bao gồm:

- Multi-Query Attention (MQA)
- Grouped Query Attention (GQA)
- Flash Attention
- Transformer-XL
- ALiBi
- Rotary Embedding
- Memory Tokens
- RMSNorm
- Residual Attention
- Talking-Heads Attention

Link:

https://github.com/lucidrains/x-transformers

---

## Large Language Models Using MQA/GQA

### [6] PaLM

Chowdhery et al.

**PaLM: Scaling Language Modeling with Pathways**

Link:

https://arxiv.org/abs/2204.02311

MQA được sử dụng để giảm KV Cache trong quá trình suy luận quy mô lớn.

---

### [7] AlphaCode

Li et al.

**Competition-Level Code Generation with AlphaCode**

Link:

https://arxiv.org/abs/2203.07814

AlphaCode là một trong những hệ thống quy mô lớn đầu tiên xác nhận hiệu quả thực tiễn của Multi-Query Attention.

---

## Recommended Reading Order

Để hiểu đầy đủ quá trình tiến hóa của Attention hiện đại:

```text
Attention Is All You Need (2017)
            │
            ▼
Multi-Head Attention
            │
            ▼
One Write-Head Is All You Need (2019)
            │
            ▼
Multi-Query Attention (MQA)
            │
            ▼
Grouped Query Attention (GQA)
            │
            ▼
PaLM / AlphaCode
            │
            ▼
LLaMA-2 / Gemma / Mistral / Gemini
```

---

## Citation

Nếu sử dụng tài liệu này trong nghiên cứu hoặc học tập:

```bibtex
@article{shazeer2019fast,
  title={Fast Transformer Decoding: One Write-Head is All You Need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
}

@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebron, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023}
}
```