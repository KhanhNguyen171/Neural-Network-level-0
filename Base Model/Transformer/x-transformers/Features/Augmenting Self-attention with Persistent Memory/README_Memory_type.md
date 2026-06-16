## Các loại Memory trong quá trình phát triển Transformer

Mặc dù tất cả các kiến trúc dưới đây đều được gọi là "Memory", nhưng bản chất của chúng rất khác nhau.

Ta có thể xem chúng như các bước tiến hóa liên tiếp nhằm giải quyết bài toán:

> Làm thế nào để Transformer nhớ được nhiều thông tin hơn ngoài context window hiện tại?

---

| Architecture            | Memory Type               |
| ----------------------- | ------------------------- |
| Transformer             | None                      |
| Relative Attention      | Positional Memory         |
| Transformer-XL          | Segment Memory            |
| Compressive Transformer | Compressed Memory         |
| Memorizing Transformer  | External Retrieval Memory |
| RETRO                   | Database Retrieval        |
| Persistent Memory       | Learned Global Memory     |

---

## 1. Transformer — No Memory

Transformer nguyên thủy không có bộ nhớ thực sự.

Nó chỉ nhìn thấy:

```text
Current Sequence

Token1 Token2 Token3 Token4
```

Sau khi Attention kết thúc:

```text
Memory = Destroyed
```

Mọi thông tin đều mất đi.

---

### Minh họa

```text
Input Sequence
      │
      ▼

+------------+
| Attention  |
+------------+

      │
      ▼

Output

(No Memory)
```

---

## 2. Relative Attention — Positional Memory

Transformer chuẩn chỉ biết:

```text
Token A
Token B
```

nhưng không biết khoảng cách giữa chúng.

Relative Attention bổ sung:

```text
Distance(A,B)
```

vào Attention Score.

---

### Memory được lưu là gì?

```text
Token Distance
```

Ví dụ:

```text
A ---- B
distance = +1

A -------- C
distance = +2
```

---

### Minh họa

```text
Token A

    │

    ▼

Relative Position

    │

    ▼

Attention Score
```

Memory ở đây không lưu nội dung.

Nó chỉ lưu:

```text
Vị trí tương đối
```

---

## 3. Transformer-XL — Segment Memory

Transformer chỉ nhìn thấy context hiện tại.

Transformer-XL giữ lại hidden states của đoạn trước.

---

### Memory được lưu là gì?

```text
Previous Hidden States
```

---

### Minh họa

```text
Segment 1

A B C D

      │
      ▼

Stored Memory

      │
      ▼

Segment 2

E F G H
```

Attention của Segment 2 có thể nhìn lại Segment 1.

---

### Ý tưởng

```text
Current Context + Previous Context
```

---

## 4. Compressive Transformer — Compressed Memory

Transformer-XL gặp vấn đề:

```text
Memory grows forever
```

Compressive Transformer nén bộ nhớ cũ.

---

### Memory được lưu là gì?

```text
Compressed Hidden States
```

---

### Minh họa

```text
Old Memory

A B C D E F G H

        │
        ▼

Compression

        │
        ▼

A' B' C'
```

---

### Ý tưởng

Giữ:

```text
Summary
```

thay vì giữ toàn bộ dữ liệu.

---

## 5. Persistent Memory — Learned Global Memory

Persistent Memory không lưu context.

Thay vào đó mô hình học trực tiếp một tập memory vectors.

---

### Memory được lưu là gì?

```text
Learnable Parameters
```

---

### Minh họa

```text
Memory Slot 1

[Knowledge]

Memory Slot 2

[Knowledge]

Memory Slot 3

[Knowledge]
```

---

Attention truy cập:

```text
Current Tokens

       +

Learned Memory
```

---

### Ý tưởng

```text
Context Retrieval + Knowledge Retrieval
```

---

## 6. Memorizing Transformer — External Retrieval Memory

Persistent Memory bị giới hạn bởi số tham số.

Memorizing Transformer tạo một cơ sở dữ liệu bên ngoài.

---

### Memory được lưu là gì?

```text
Past Activations
```

---

### Minh họa

```text
Current Query

      │
      ▼

Nearest Neighbor Search

      │
      ▼

Past Tokens
```

---

Mô hình có thể truy xuất:

```text
Millions of previous tokens
```

mà không cần lưu trong context.

---

## 7. RETRO — Database Retrieval

RETRO tiến thêm một bước.

Thay vì lưu hidden states:

nó lưu trực tiếp dữ liệu huấn luyện.

---

### Memory được lưu là gì?

```text
External Text Database
```

---

### Minh họa

```text
Question

    │
    ▼

Database Search

    │
    ▼

Relevant Documents

    │
    ▼

Attention
```

---

### Ý tưởng

```text
LLM + Search Engine
```

---

## So sánh trực quan

```text
Transformer

No Memory
───────────────

Relative Attention

Position Memory
───────────────

Transformer-XL

Past Context
───────────────

Compressive Transformer

Compressed Context
───────────────

Persistent Memory

Learned Knowledge
───────────────

Memorizing Transformer

External Token Memory
───────────────

RETRO

External Document Database
```

---

## Phân loại bản chất bộ nhớ

### Position Memory

```text
Relative Attention
```

Lưu:

```text
Khoảng cách giữa token
```

---

### Context Memory

```text
Transformer-XL
Compressive Transformer
```

Lưu:

```text
Hidden States của các đoạn trước
```

---

### Parameter Memory

```text
Persistent Memory
```

Lưu:

```text
Tri thức bên trong tham số mô hình
```

---

### External Memory

```text
Memorizing Transformer
RETRO
```

Lưu:

```text
Tri thức bên ngoài mô hình
```

---

## Góc nhìn tiến hóa

```text
No Memory
     │
     ▼

Position Memory
     │
     ▼

Context Memory
     │
     ▼

Compressed Memory
     │
     ▼

Learned Global Memory
     │
     ▼

External Memory
     │
     ▼

Retrieval-Augmented LLM
```

Đây chính là con đường phát triển dẫn từ Transformer nguyên thủy đến các hệ thống bộ nhớ hiện đại trong các LLM quy mô lớn.
