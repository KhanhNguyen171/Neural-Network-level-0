# Concatenate Memory Tokens trong Transformer

> Hiểu trực quan cách các kiến trúc Memory Transformer đưa bộ nhớ vào Attention Network.

---

# 1. Vấn đề của Transformer chuẩn

Transformer nguyên thủy chỉ chứa các token dữ liệu:

```text
x1  x2  x3  x4
```

Self-Attention cho phép mọi token giao tiếp với nhau:

```text
x1 ←→ x2
↑  ╲ ╱ ↑
│   X  │
↓ ╱ ╲ ↓
x3 ←→ x4
```

Thông tin được lưu trực tiếp bên trong các token dữ liệu.

---

## Hạn chế

Không tồn tại vùng nhớ riêng:

```text
Data Token
     =
Computation
     +
Storage
```

Mọi token vừa phải:

* biểu diễn dữ liệu
* lưu trạng thái
* truyền thông tin

đồng thời.

---

# 2. Ý tưởng Concatenate Memory

Thay vì:

```text
[x1 x2 x3 x4]
```

ta thêm các token bộ nhớ:

```text
[m1 m2 m3]
```

rồi nối vào chuỗi.

```text
[m1 m2 m3 x1 x2 x3 x4]
```

Đây chính là:

```python
X = torch.cat([memory_tokens, input_tokens], dim=1)
```

---

## Concatenate nghĩa là gì?

Nếu:

```text
Memory

m1 m2 m3
```

và

```text
Input

x1 x2 x3 x4
```

thì:

```text
Concatenate

[m1 m2 m3 x1 x2 x3 x4]
```

---

## Kích thước tensor

Trước:

```text
Input

(n × d)

4 × 512
```

Sau:

```text
Memory

(k × d)

3 × 512
```

```text
Concatenate

(k+n) × d

7 × 512
```

---

# 3. Transformer Layer Chuẩn

## Pipeline

```text
Input Tokens

x1 x2 x3 x4

      │
      ▼

Multi Head Attention

      │
      ▼

Residual

      │
      ▼

Feed Forward

      │
      ▼

Output
```

---

## Luồng Attention

```text
x1 ↔ x2
x1 ↔ x3
x1 ↔ x4

x2 ↔ x3
x2 ↔ x4

x3 ↔ x4
```

Mọi token kết nối trực tiếp với nhau.

---

# 4. Memory Transformer (MemTransformer)

## Ý tưởng

Thêm Memory Tokens.

```text
[m1 m2 m3 x1 x2 x3 x4]
```

sau đó đưa nguyên chuỗi vào Transformer.

---

## Pipeline

```text
Memory Tokens

m1 m2 m3

      │

Input Tokens

x1 x2 x3 x4

      │

      ▼

Concatenate

[m1 m2 m3 x1 x2 x3 x4]

      │

      ▼

Transformer Layer

      │

      ▼

Output
```

---

## Attention

Memory và Data hoàn toàn bình đẳng.

```text
m1 ↔ m2
m1 ↔ m3

m1 ↔ x1
m1 ↔ x2
m1 ↔ x3
m1 ↔ x4

...

x1 ↔ x2
x1 ↔ x3
```

---

## Điều quan trọng

Transformer KHÔNG biết:

```text
m1 là memory
x1 là data
```

Nó chỉ thấy:

```text
Sequence Length = k + n
```

và học tự động vai trò của các token.

---

# 5. MemCtrl Transformer

## Ý tưởng

Memory không còn tự tổ chức hoàn toàn.

Thêm một mạng chuyên điều khiển bộ nhớ.

```text
Memory Controller
```

---

## Pipeline

```text
Input Tokens

x1 x2 x3 x4

      │

      ▼

Memory Controller

      │

      ▼

Memory Tokens

m1 m2 m3

      │

      ▼

Transformer
```

---

## Trực quan

Transformer chuẩn:

```text
Tokens
   │
   ▼
Attention
```

MemCtrl:

```text
Tokens
   │
   ▼

Controller

   │
   ▼

Memory

   │
   ▼

Attention
```

---

## Vai trò

Controller học:

```text
Ghi cái gì?

Xóa cái gì?

Giữ cái gì?
```

trong memory.

---

# 6. Memory Bottleneck Transformer

Đây là biến thể quan trọng nhất.

---

## Mục tiêu

Không cho token dữ liệu giao tiếp trực tiếp.

Buộc mọi thông tin phải đi qua memory.

---

# Transformer chuẩn

```text
x1 ←→ x2
↑  ╲ ╱ ↑
│   X  │
↓ ╱ ╲ ↓
x3 ←→ x4
```

Kết nối:

[
O(n^2)
]

---

# Memory Bottleneck

```text
          m1 m2 m3
         / | | \
        /  | |  \
       /   | |   \

     x1   x2 x3  x4
```

---

Không tồn tại:

```text
x1 ↔ x2
x1 ↔ x3
x1 ↔ x4
```

---

Mọi thông tin phải đi qua:

```text
Memory Tokens
```

---

# Bước 1

## Cập nhật Memory

Memory nhìn thấy toàn bộ chuỗi.

```text
Memory

m1 m2 m3

     ▲
     │

x1 x2 x3 x4
```

---

Pipeline

```text
Input + Memory

[m1 m2 m3 x1 x2 x3 x4]

         │

         ▼

Update Memory
```

---

Sau bước này:

```text
m1'
m2'
m3'
```

chứa thông tin toàn cục.

---

# Bước 2

## Cập nhật Data

Token dữ liệu chỉ nhìn thấy Memory.

```text
m1'
m2'
m3'

  ▲
  │

x1
x2
x3
x4
```

---

Pipeline

```text
Updated Memory

m1' m2' m3'

        │

        ▼

Update Data Tokens

x1 x2 x3 x4
```

---

# Luồng thông tin

Transformer chuẩn:

```text
x1
 │
 ▼

x2
 │
 ▼

x3
 │
 ▼

x4
```

---

Memory Bottleneck:

```text
x1
 │
 ▼

Memory

 │
 ▼

x4
```

---

Hay:

```text
Data
  │
  ▼

Memory

  │
  ▼

Data
```

---

# Tại sao gọi là Bottleneck?

Vì toàn bộ thông tin bắt buộc phải đi qua:

```text
m1 m2 m3
```

giống như cổ chai:

```text
Wide Flow

██████████

     │

     ▼

███

     │

     ▼

██████████
```

---

Memory trở thành:

```text
Information Compression Layer
```

---

# So sánh các kiến trúc

| Kiến trúc      | Memory              | Attention                     |
| -------------- | ------------------- | ----------------------------- |
| Transformer    | Không có            | Data ↔ Data                   |
| MemTransformer | Memory Tokens       | Memory ↔ Data                 |
| MemCtrl        | Memory + Controller | Controller điều khiển Memory  |
| MemBottleneck  | Memory Bottleneck   | Data chỉ giao tiếp qua Memory |

---

# Tổng kết

Có thể hiểu quá trình tiến hóa như sau:

```text
Transformer

x ↔ x
```

↓

```text
MemTransformer

x ↔ x
↕
m
```

↓

```text
MemCtrl

x ↔ x
↕
Controller
↕
m
```

↓

```text
Memory Bottleneck

x
│
▼

m

▲
│

x
```

Tư tưởng trung tâm của các kiến trúc này là:

```text
Tách riêng

Storage
    khỏi
Computation
```

bằng cách đưa các Memory Tokens vào bên trong Attention Network và biến chúng thành một vùng nhớ chuyên dụng cho Transformer.
