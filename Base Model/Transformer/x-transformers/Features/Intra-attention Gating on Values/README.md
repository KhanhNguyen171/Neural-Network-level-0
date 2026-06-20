# Intra-attention Gating on Values

> **Nguồn tham khảo**
>
> - x-transformers (lucidrains): https://github.com/lucidrains/x-transformers
> - AlphaFold2 (DeepMind): https://github.com/google-deepmind/alphafold
>
> Ý tưởng được mô tả trong x-transformers như một biến thể attention lấy cảm hứng từ cơ chế attention trong AlphaFold2:
>
> > "gate aggregated values with the input"

---

# 1. Tổng quan

**Intra-attention Gating on Values** là một mở rộng của cơ chế Multi-Head Self-Attention (MSA), trong đó:

- Attention vẫn tính toán bình thường.
- Các giá trị đã được tổng hợp từ attention (**aggregated values**) không được đưa thẳng tới output projection.
- Thay vào đó chúng được **gating (điều tiết)** bởi chính biểu diễn đầu vào.

Mục tiêu:

- Cho attention block quyền kiểm soát mạnh hơn đối với phần thông tin được ghi trở lại residual stream.
- Giảm hiện tượng cập nhật quá mức (over-update).
- Tạo cơ chế tương tự một bộ lọc động (dynamic filter) cho output của attention.

Trong x-transformers:

```python
Encoder(
    dim = 512,
    depth = 6,
    heads = 8,
    attn_gate_values = True
)
```

---

# 2. Động cơ khoa học

## 2.1 Attention chuẩn

Attention chuẩn:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

trong đó:

$$
X \in \mathbb{R}^{n \times d}
$$

Attention scores:

$$
A = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_h}} \right)
$$

Output attention:

$$
O = AV
$$

Sau đó:

$$
Y = OW_O
$$

Toàn bộ thông tin từ \(O\) sẽ được ghi vào residual stream.

---

## 2.2 Vấn đề

Trong thực tế:

$$
O = AV
$$

có thể chứa:

- thông tin hữu ích
- thông tin nhiễu
- cập nhật không cần thiết

Attention không có cơ chế trực tiếp để quyết định:

> "Bao nhiêu phần của output này nên được ghi lại?"

Nói cách khác:

Attention quyết định **where to read** nhưng chưa thực sự quyết định **how much to write**.

---

# 3. Ý tưởng chính

Thay vì dùng:

$$
O = AV
$$

ta xây dựng một gate:

$$
G = \sigma(XW_G)
$$

với:

$$
\sigma = \text{Sigmoid}
$$

Gate:

$$
G \in [0,1]
$$

sau đó:

$$
O_{gate} = G \odot O
$$

Trong đó:

$$
\odot
$$

là nhân từng phần tử (element-wise multiplication).

---

# 4. Trực giác

Attention chuẩn:

```text
Attention Output
       │
       ▼
Residual Stream
```

Intra-attention Gating:

```text
Attention Output
       │
       ▼
     Gate
       ▲
       │
     Input
       │
       ▼
Residual Stream
```

Input quyết định:

- phần nào của attention output được giữ lại
- phần nào bị triệt tiêu

---

# 5. Kiến trúc toán học đầy đủ

## Bước 1

Sinh Query, Key, Value

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

---

## Bước 2

Attention map

$$
A = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_h}} \right)
$$

---

## Bước 3

Aggregated Values

$$
O = AV
$$

---

## Bước 4

Sinh Gate từ Input

$$
G = \sigma(XW_G)
$$

---

## Bước 5

Gate Output

$$
O_{gate} = G \odot O
$$

---

## Bước 6

Output Projection

$$
Y = O_{gate}W_O
$$

---

# 6. Minh họa toàn bộ luồng dữ liệu

```text
Input X
   │
   ├───────────────┐
   │               │
   ▼               ▼
Query          Gate Projection
Key                │
Value              ▼
   │           Sigmoid
   ▼               │
Attention          ▼
   │            Gate G
   ▼               │
Aggregated O ◄─────┘
   │
   ▼
O_gate = G ⊙ O
   │
   ▼
Output Projection
   │
   ▼
Residual Update
```

---

# 7. Liên hệ với hình AlphaFold2

Kiến trúc được sử dụng trong AlphaFold2 gần giống:

```text
Input Representation
         │
         ├─────────────► Gate
         │
         ▼
     Attention
         │
         ▼
   Aggregated Values
         │
         ▼
    Multiply
         │
         ▼
      Output
```

Ý tưởng:

Input ban đầu kiểm soát trực tiếp lượng thông tin được ghi trở lại.

Đây là một dạng:

$$
\text{Read} \rightarrow \text{Filter} \rightarrow \text{Write}
$$

thay vì:

$$
\text{Read} \rightarrow \text{Write}
$$

---

# 8. So sánh với Gated Linear Unit (GLU)

GLU:

$$
\text{GLU}(x) = (Ax) \odot \sigma(Bx)
$$

Intra-attention Gating:

$$
O_{gate} = (AV) \odot \sigma(XW_G)
$$

Giống nhau:

- đều dùng sigmoid gate
- đều nhân element-wise

Khác nhau:

GLU gate feed-forward output.

Intra-attention gate attention output.

---

# 9. So sánh với Attention on Attention

Attention-on-Attention (AoA):

```text
Attention
     │
     ▼
Extra Attention Layer
     │
     ▼
Output
```

Intra-attention Gating:

```text
Attention
     │
     ▼
Gate
     │
     ▼
Output
```

AoA:

- tăng khả năng biểu diễn
- tăng số phép tính

Gating:

- rẻ hơn
- ít tham số hơn
- dễ huấn luyện hơn

---

# 10. Góc nhìn Information Flow

Attention chuẩn:

$$
X \rightarrow AV \rightarrow Residual
$$

Gated Attention:

$$
X \rightarrow AV
$$

$$
X \rightarrow G
$$

$$
AV \rightarrow G \odot AV \rightarrow Residual
$$

Do đó block có thể học:

- ghi mạnh
- ghi yếu
- không ghi

cho từng chiều embedding.

---

# 11. Góc nhìn Residual Update

Transformer block thực chất học:

$$
x_{l+1} = x_l + \Delta x_l
$$

với:

$$
\Delta x_l = AVW_O
$$

Khi thêm gate:

$$
\Delta x_l = (G \odot AV)W_O
$$

Gate trở thành:

$$
\text{Write Controller}
$$

quyết định độ lớn của:

$$
\Delta x_l
$$

---

# 12. Chi phí tính toán

Attention chuẩn:

$$
O(n^2 d)
$$

Gated Attention:

thêm:

$$
XW_G
$$

chi phí:

$$
O(nd^2)
$$

nhỏ hơn đáng kể so với chi phí attention khi:

$$
n \gg d
$$

Do đó gần như không làm tăng đáng kể FLOPs.

---

# 13. Vai trò trong hệ sinh thái x-transformers

Trong x-transformers, cơ chế này thuộc nhóm:

```text
Attention Refinements
```

bao gồm:

- Talking-Heads Attention
- Attention on Attention
- Residual Attention
- Memory Tokens
- Gated Values Attention
- Dynamic Position Bias
- ALiBi
- Rotary Embedding

Mục tiêu chung:

> cải thiện chất lượng attention mà không thay đổi cấu trúc Transformer nền tảng.

---

# 14. Pseudocode

```python
def gated_attention(x):

    q = Wq(x)
    k = Wk(x)
    v = Wv(x)

    attn = softmax(q @ k.T / sqrt(d))

    out = attn @ v

    gate = sigmoid(Wg(x))

    out = out * gate

    return Wo(out)
```

---

# 15. Sơ đồ tổng kết

```text
                 INTRA-ATTENTION GATING

                 ┌─────────────┐
                 │   Input X   │
                 └──────┬──────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼

  Multi-Head Attention          Gate Projection

  Q = XWQ                       G = σ(XWG)
  K = XWK
  V = XWV

        │                               │
        ▼                               ▼

     O = AV                        Gate G

        └───────────────┬───────────────┘
                        ▼

                 O_gate = G ⊙ O

                        │
                        ▼

                 Output Projection

                        │
                        ▼

                 Residual Update
```

---

# Kết luận

Intra-attention Gating on Values bổ sung một **cơ chế kiểm soát ghi (write-control mechanism)** vào Self-Attention. Thay vì ghi trực tiếp aggregated values vào residual stream, mô hình học một gate phụ thuộc vào đầu vào:

$$
G=\sigma(XW_G)
$$

và thực hiện:

$$
O_{gate} = G \odot (AV)
$$

Điều này biến attention từ một cơ chế chỉ quyết định **đọc ở đâu (where to read)** thành một cơ chế đồng thời quyết định **ghi bao nhiêu (how much to write)**. Đây chính là lý do biến thể này xuất hiện trong AlphaFold2 và sau đó được đưa vào x-transformers như một cải tiến nhẹ nhưng hiệu quả cho kiến trúc Transformer hiện đại.