# Transformer Memory Evolution

> Phụ lục mô tả các loại bộ nhớ (Memory Types) xuất hiện trong quá trình phát triển của Transformer.

---

# Tổng quan

Mọi kiến trúc Transformer đều cố gắng giải quyết cùng một bài toán:

> Làm thế nào để mô hình ghi nhớ được nhiều thông tin hơn ngoài context hiện tại?

Các thế hệ Transformer khác nhau chủ yếu khác nhau ở:

* Vị trí lưu trữ thông tin
* Thời gian tồn tại của thông tin
* Cách truy xuất thông tin

---

# 1. Vanilla Transformer

## Memory Type

```text
None
```

## Kiến trúc

```text
x1 x2 x3 x4

 │  │  │  │

 ▼  ▼  ▼  ▼

Self Attention

 │  │  │  │

 ▼  ▼  ▼  ▼

Output
```

## Đặc điểm

Transformer nguyên thủy không có bộ nhớ riêng.

Toàn bộ thông tin chỉ tồn tại trong:

```text
Token Embeddings
```

Mỗi token vừa:

* chứa dữ liệu
* chứa trạng thái tính toán

## Hạn chế

Khi context biến mất:

```text
Previous Context
      ❌
```

toàn bộ thông tin cũng mất theo.

---

# 2. Relative Attention

## Memory Type

```text
Positional Memory
```

## Kiến trúc

```text
Token A
   │

Relative Distance

   │

Token B
```

## Ý tưởng

Không lưu nội dung.

Chỉ lưu:

```text
Khoảng cách tương đối
```

giữa các token.

Ví dụ:

```text
A ← 3 → B
```

## Thông tin được nhớ

```text
Where?
```

Không phải:

```text
What?
```

---

# 3. Transformer-XL

## Memory Type

```text
Segment Memory
```

## Kiến trúc

```text
Segment 1

x1 x2 x3 x4

        │
        ▼

Stored Memory

        │
        ▼

Segment 2

x5 x6 x7 x8
```

## Ý tưởng

Lưu hidden states của đoạn trước.

```text
Past Hidden States
```

được đưa vào attention của đoạn kế tiếp.

## Thông tin được nhớ

```text
Previous Context
```

## Lợi ích

Tăng context dài hơn giới hạn của một segment.

---

# 4. Compressive Transformer

## Memory Type

```text
Compressed Memory
```

## Kiến trúc

```text
Recent Memory

x1 x2 x3 x4

        │
        ▼

Compression

        │
        ▼

Compressed Memory
```

## Ý tưởng

Thông tin cũ không bị xóa.

Thay vào đó:

```text
Nén lại
```

rồi lưu tiếp.

## Thông tin được nhớ

```text
Long-Term Context
```

---

# 5. Memorizing Transformer

## Memory Type

```text
Retrieval Memory
```

## Kiến trúc

```text
Current Tokens

       │

       ▼

Nearest Neighbor Search

       │

       ▼

External Memory
```

## Ý tưởng

Thay vì nhớ mọi thứ trong mô hình.

Transformer tìm kiếm:

```text
KNN Database
```

ở bên ngoài.

## Thông tin được nhớ

```text
Past Examples
```

---

# 6. RETRO

## Memory Type

```text
External Database
```

## Kiến trúc

```text
Input

  │

  ▼

Retriever

  │

  ▼

Database Chunks

  │

  ▼

Transformer
```

## Ý tưởng

Transformer truy xuất dữ liệu từ cơ sở tri thức ngoài mô hình.

## Thông tin được nhớ

```text
External Knowledge
```

---

# 7. Memory Transformer

## Memory Type

```text
Learnable Memory Tokens
```

## Kiến trúc

```text
m1 m2 m3

 │  │  │

 ▼  ▼  ▼

x1 x2 x3 x4
```

## Ý tưởng

Bổ sung:

```text
Memory Tokens
```

vào chuỗi attention.

## Vai trò

Memory token là vùng nhớ toàn cục của mạng.

## Thông tin được nhớ

```text
Global Context
```

---

# 8. Register Transformer

## Memory Type

```text
Register Tokens
```

## Kiến trúc

```text
Register Tokens

      │

      ▼

Attention Network
```

## Ý tưởng

Tạo các token chuyên dùng để lưu trạng thái trung gian.

## Mục tiêu

Giảm:

```text
Activation Outliers
```

trong attention.

---

# 9. Persistent Memory Transformer

## Memory Type

```text
Persistent Learned Memory
```

## Kiến trúc

```text
Persistent Memory

p1 p2 p3 p4

      │

      ▼

Attention
```

## Ý tưởng

Memory không còn là bộ nhớ tạm.

Nó trở thành:

```text
Learned Knowledge
```

được huấn luyện xuyên suốt toàn bộ quá trình training.

## Vai trò

Tương tự:

```text
Neural Database
```

bên trong mô hình.

---

# 10. Hymba

## Memory Type

```text
Meta Tokens
```

## Kiến trúc

```text
Past Tokens

      │

Meta Tokens

      │

Current Token

      │

Prediction
```

## Ý tưởng

Meta token hoạt động như vùng nhớ cố định trong mô hình autoregressive.

## Thông tin được nhớ

```text
Generation State
```

---

# Sơ đồ tiến hóa Memory

```text
Transformer
    │
    ▼

Relative Attention
(Positional Memory)

    │
    ▼

Transformer-XL
(Segment Memory)

    │
    ▼

Compressive Transformer
(Compressed Memory)

    │
    ▼

Memorizing Transformer
(Retrieval Memory)

    │
    ▼

RETRO
(External Database)

    │
    ▼

Memory Transformer
(Memory Tokens)

    │
    ▼

Register Transformer
(Register Tokens)

    │
    ▼

Persistent Memory
(Persistent Memory)

    │
    ▼

Hymba
(Meta Tokens)
```

---

# Bảng tổng hợp

| Architecture            | Memory Type               | Lưu ở đâu?         | Nhớ cái gì?           |
| ----------------------- | ------------------------- | ------------------ | --------------------- |
| Transformer             | None                      | Không có           | Không có              |
| Relative Attention      | Positional Memory         | Attention Bias     | Vị trí                |
| Transformer-XL          | Segment Memory            | Hidden States      | Context trước         |
| Compressive Transformer | Compressed Memory         | Compressed States  | Context dài           |
| Memorizing Transformer  | Retrieval Memory          | KNN Store          | Ví dụ quá khứ         |
| RETRO                   | External Database         | Database           | Tri thức ngoài        |
| Memory Transformer      | Memory Tokens             | Attention Tokens   | Context toàn cục      |
| Register Transformer    | Register Tokens           | Attention Tokens   | Trạng thái trung gian |
| Persistent Memory       | Persistent Learned Memory | Learned Parameters | Tri thức học được     |
| Hymba                   | Meta Tokens               | Attention Tokens   | Trạng thái sinh token |

---

# Tư tưởng cốt lõi

Toàn bộ quá trình tiến hóa có thể được nhìn nhận như:

```text
No Memory
     │
     ▼

Positional Memory
     │
     ▼

Temporal Memory
     │
     ▼

Compressed Memory
     │
     ▼

Retrieval Memory
     │
     ▼

Learnable Memory
     │
     ▼

Persistent Memory
```

Hay:

```text
Transformer
      ↓

Remember Position
      ↓

Remember Context
      ↓

Remember History
      ↓

Remember Knowledge
      ↓

Learn Internal Memory
```

Đây chính là tuyến phát triển dẫn tới các kiến trúc bộ nhớ hiện đại trong hệ sinh thái x-transformers và các Large Language Models thế hệ mới.
