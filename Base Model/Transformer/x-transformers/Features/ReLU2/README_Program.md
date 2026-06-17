# Phụ lục B — Ví dụ về Program Representation trong DNA Search Space

## B.1 Mục đích của hình

Hình này minh họa cách một **Subprogram** trong hệ thống DNA được biểu diễn đồng thời dưới ba dạng:

1. Chuỗi Instructions (Program Representation)
2. TensorFlow Code (Executable Representation)
3. Computational Graph (Graph Representation)

Ba biểu diễn này thực chất là ba góc nhìn khác nhau của cùng một kiến trúc.

```text id="9ep08z"
Program
    ↓
TensorFlow Code
    ↓
Computation Graph
```

Mục tiêu của DNA là cho phép Neural Architecture Search thao tác trực tiếp trên biểu diễn chương trình thay vì thao tác trên đồ thị mạng thần kinh truyền thống.

---

# B.2 Cấu trúc tổng thể

Subprogram trong hình gồm bốn instruction:

```text id="8g5prn"
(h2) CONV1X1
(h3) SIN
(h4) MUL
(h5) ADD
```

Mỗi instruction tạo ra một hidden state mới.

```text id="v3f5lm"
h0
h1
 │
 ▼
Instructions
 │
 ▼
h2
h3
h4
h5
```

Trong DNA:

$$
h_i
$$

được xem là các thanh ghi trung gian (intermediate registers) tương tự như biến trong một chương trình máy tính.

---

# B.3 Input States

Hai đầu vào ban đầu:

```text id="c6hz65"
(h0)
(h1)
```

được cung cấp cho subprogram.

Ta có thể xem:

$$
h_0
$$

và

$$
h_1
$$

là hai tensor nguồn.

Ví dụ trong Transformer:

$$
h_0= \text{hidden state}
$$

$$
h_1= \text{context state}
$$

Tuy nhiên trong DNA chúng chỉ được xem là các tensor tổng quát.

---

# B.4 Instruction 1 — CONV 1×1

## Định nghĩa

Instruction đầu tiên:

```text id="3r18qv"
(h2) CONV1X1
```

nhận đầu vào:

```text id="0m1r3v"
Input_1 = h0
Dim = 512
```

và tạo:

$$
h_2
$$

---

## TensorFlow tương ứng

Trong hình:

```python id="qrrikm"
h2 = tf.layers.dense(h0, 512)
```

Điều này cho thấy:

```text id="ptd5ll"
CONV1X1
```

được ánh xạ thành:

```text id="h8xjli"
Linear Projection
```

hay:

$$
h_2 = W h_0 + b
$$

---

## Ý nghĩa

Trong Transformer hiện đại:

```text id="n9k3pn"
Dense Layer
Projection Layer
Feed Forward Expansion
```

đều thuộc nhóm toán tử này.

---

# B.5 Instruction 2 — SIN

Instruction thứ hai:

```text id="0gng3o"
(h3) SIN
```

nhận đầu vào:

$$
h_1
$$

và sinh:

$$
h_3
$$

---

## TensorFlow tương ứng

```python id="r6sdlh"
h3 = tf.math.sin(h1)
```

---

## Ý nghĩa

Đây là một toán tử phi tuyến:

$$
h_3= \sin(h_1)
$$

DNA cho phép sử dụng nhiều primitive khác nhau:

```text id="7s8r1u"
SIN
COS
TANH
MAX
MIN
ADD
MUL
```

để mở rộng không gian tìm kiếm.

Mục tiêu là cho phép NAS tự quyết định dạng phi tuyến nào phù hợp.

---

# B.6 Instruction 3 — MUL

Instruction tiếp theo:

```text id="0ebg89"
(h4) MUL
```

sử dụng:

```text id="yslkkh"
Input_1 = h2
Input_2 = h3
```

---

## TensorFlow tương ứng

```python id="6oncv9"
h4 = tf.math.multiply(h2, h3)
```

---

## Phương trình

$$
h_4= h_2 \odot h_3
$$

trong đó:

$$
\odot
$$

là phép nhân từng phần tử (element-wise multiplication).

---

## Vai trò

Đây chính là cơ chế nền tảng của:

```text id="l6jv6v"
GLU
GEGLU
SwiGLU
Gating Networks
Attention Masks
```

Mặc dù hình minh họa không mô tả GLU trực tiếp, nhưng phép nhân tensor là cơ chế cốt lõi của mọi kiến trúc gating hiện đại.

---

# B.7 Instruction 4 — ADD

Instruction cuối cùng:

```text id="2z7dzr"
(h5) ADD
```

sử dụng:

```text id="11hphz"
Input_1 = h0
Input_2 = h2
```

---

## TensorFlow tương ứng

```python id="wpwldw"
h5 = tf.math.add(h0, h2)
```

---

## Phương trình

$$
h_5= h_0 + h_2
$$

---

## Ý nghĩa

Đây chính là một dạng:

```text id="ejsaew"
Residual Connection
Skip Connection
```

được sử dụng rộng rãi trong:

```text id="1lvt7j"
ResNet
Transformer
GPT
BERT
T5
```

---

# B.8 Output của Subprogram

Dòng cuối:

```python id="j7xyrq"
output = h5
```

cho thấy:

$$
Output = h_5
$$

---

Toàn bộ Subprogram được biểu diễn:

$$
(h_0,h_1) \rightarrow h_5
$$

---

# B.9 Chuyển đổi sang Computation Graph

Sau khi biên dịch, chương trình trở thành đồ thị tính toán.

```text id="54v63i"
h0 ──► CONV1X1 ──► h2 ──┐
 │                       │
 │                       ▼
 └──────────────► ADD ─► OUT

h1 ──► SIN ──────► h3
                   │
                   ▼
                 MUL
```

Điều quan trọng là:

> Đồ thị này không được thiết kế thủ công.

Nó được sinh tự động từ chương trình DNA.

---

# B.10 Tại sao dùng Program Representation?

Trong NAS truyền thống:

```text id="4r7x9w"
Node
Edge
Node
Edge
...
```

không gian tìm kiếm rất lớn.

---

DNA chuyển bài toán thành:

```text id="b55f24"
Instruction
Instruction
Instruction
...
```

giống như sinh mã nguồn.

Điều này giúp:

* tái sử dụng module
* tạo cấu trúc phân cấp
* dễ đột biến (mutation)
* dễ lai ghép (crossover)

trong quá trình tiến hóa kiến trúc.

---

# B.11 Quan hệ với Transformer

Một Transformer layer có thể được xem như:

```text id="vqccrj"
Projection
Activation
Residual
Normalization
Attention
```

Tất cả đều có thể được biểu diễn bằng instruction.

Ví dụ:

```text id="mkekv6"
Dense
ReLU²
Add
LayerNorm
```

chỉ là các primitive khác nhau trong không gian DNA.

Do đó NAS có khả năng phát hiện:

```text id="7s0msm"
Squared ReLU
```

hay:

```text id="7l2m3e"
Multi-DConv Head Attention
```

bằng cách tái tổ hợp các instruction này.

---

# B.12 Góc nhìn Compiler

Một cách nhìn chính xác hơn là:

```text id="vh7h9g"
DNA
=
Neural Architecture Programming Language
```

Trong đó:

```text id="i4h1yv"
Instruction
```

tương đương:

```text id="3hv3sm"
Machine Instruction
```

---

```text id="6ax8vx"
Subprogram
```

tương đương:

```text id="lk7j1r"
Function
```

---

```text id="1xkrjk"
Generated TF Code
```

tương đương:

```text id="j3qtx0"
Compiled Program
```

---

```text id="7v7c9n"
Neural Network
```

tương đương:

```text id="42zqaf"
Executable Binary
```

---

# B.13 Ý nghĩa học thuật

Hình này minh họa ý tưởng trung tâm của DNA:

$$
\boxed{ Neural\ Architecture= Program }
$$

thay vì:

$$
\boxed{ Neural\ Architecture= Graph}
$$

Sự thay đổi góc nhìn này cho phép Neural Architecture Search khai thác các kỹ thuật của:

* Program Synthesis
* Genetic Programming
* Evolutionary Computation
* Compiler Optimization

để khám phá các kiến trúc mới.

Chính cơ chế này đã dẫn đến việc phát hiện các thành phần quan trọng của Primer như **Squared ReLU** và **Multi-DConv Head Attention**, sau đó được tích hợp vào các Transformer hiện đại và thư viện x-transformers.

---

# Tóm tắt

```text id="pmhlan"
Primitive Operations
        │
        ▼
Instructions
        │
        ▼
Subprogram
        │
        ▼
TensorFlow Code
        │
        ▼
Computation Graph
        │
        ▼
Trainable Neural Network
```

Hình minh họa cách DNA biến kiến trúc mạng thần kinh thành một chương trình có thể tiến hóa, biên dịch và tối ưu hóa tự động, tạo nền tảng cho việc khám phá các kiến trúc Transformer hiệu quả hơn so với thiết kế thủ công truyền thống.
