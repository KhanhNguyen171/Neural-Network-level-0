# GPU Memory Hierarchy cho Flash Attention

> Phụ lục giải thích kiến trúc bộ nhớ GPU phục vụ việc hiểu Flash Attention và các Transformer hiện đại.

---

# Mục lục

1. Tại sao cần hiểu GPU Memory Hierarchy?
2. Các tầng bộ nhớ trong GPU
3. HBM (High Bandwidth Memory)
4. SRAM (Shared Memory)
5. Tensor Core
6. Memory Bottleneck trong Transformer
7. Vì sao Flash Attention tập trung vào IO?
8. Data Movement trong Standard Attention
9. Data Movement trong Flash Attention
10. Tổng kết

---

# 1. Tại sao cần hiểu GPU Memory Hierarchy?

Khi mới học Deep Learning, nhiều người nghĩ rằng:

> Transformer chậm vì phải thực hiện quá nhiều phép nhân ma trận.

Điều này chỉ đúng một phần.

Trong GPU hiện đại:

$$
T_{total}=T_{compute} + T_{memory}
$$

Trong nhiều trường hợp:

$$
T_{memory} \gt T_{compute}
$$

Nghĩa là:

GPU không bị giới hạn bởi số phép toán.

GPU bị giới hạn bởi tốc độ di chuyển dữ liệu.

---

# 2. Các tầng bộ nhớ trong GPU

GPU hiện đại có nhiều tầng bộ nhớ.

```text
                 GPU

 ┌───────────────────────────┐
 │       Tensor Cores        │
 │     Matrix Compute Unit   │
 └─────────────┬─────────────┘
               │
               ▼

 ┌───────────────────────────┐
 │       SRAM / Shared       │
 │      On-Chip Memory       │
 └─────────────┬─────────────┘
               │
               ▼

 ┌───────────────────────────┐
 │            HBM            │
 │     High Bandwidth RAM    │
 └───────────────────────────┘
```

---

Tốc độ truy cập:

```text
SRAM
  ▲
  │ rất nhanh
  │
  │
HBM
  ▼
 chậm hơn
```

---

# 3. HBM (High Bandwidth Memory)

HBM là bộ nhớ chính của GPU.

Có thể xem tương tự như:

```text
RAM của CPU
```

nhưng có băng thông cực lớn.

---

## Đặc điểm

Dung lượng:

```text
40GB
80GB
120GB
```

Ví dụ:

```text
A100  : 40GB / 80GB
H100  : 80GB
B200  : >100GB
```

---

HBM chứa:

* Model weights
* Activations
* Attention matrices
* Gradients
* Optimizer states

---

## Ưu điểm

Dung lượng lớn.

```text
HBM
┌────────────────────────┐
│  Hàng chục GB dữ liệu  │
└────────────────────────┘
```

---

## Nhược điểm

Khoảng cách tới Tensor Core xa hơn SRAM.

Do đó:

```text
Latency cao hơn
```

---

Minh họa:

```text
HBM

┌───────────────────┐
│   Q,K,V Tensors   │
│   Model Weights   │
│   Activations     │
└───────────────────┘

          │
          │
          ▼

Tensor Core
```

Mỗi lần đọc dữ liệu từ HBM đều tốn chi phí.

---

# 4. SRAM (Shared Memory)

SRAM nằm bên trong Streaming Multiprocessor (SM).

Nó gần Tensor Core hơn rất nhiều.

---

Minh họa:

```text
         SM

 ┌───────────────────┐
 │   Tensor Core     │
 └─────────┬─────────┘
           │
           ▼

 ┌───────────────────┐
 │       SRAM        │
 │   Shared Memory   │
 └───────────────────┘
```

---

## Đặc điểm

Dung lượng nhỏ:

```text
64 KB
96 KB
128 KB
192 KB
```

---

Nhưng:

```text
Latency cực thấp
```

và

```text
Bandwidth cực lớn
```

---

## Vai trò

SRAM giống như:

```text
CPU Cache
```

trong CPU.

---

Mục tiêu:

```text
Load dữ liệu từ HBM

↓

Lưu trong SRAM

↓

Tái sử dụng nhiều lần
```

---

# 5. Tensor Core

Tensor Core là phần cứng thực hiện:

$$
C=A\times B
$$

cực nhanh.

---

Minh họa:

```text
Tensor Core

      A
      │
      ▼

   Matrix
 Multiply

      ▲
      │
      B

      │
      ▼

      C
```

---

Tensor Core rất nhanh.

Thường nhanh hơn khả năng cung cấp dữ liệu từ HBM.

---

Điều này tạo ra:

```text
Compute Starvation
```

---

# 6. Memory Bottleneck trong Transformer

Attention truyền thống:

$$
QK^T
$$

tạo ra:

$$
N \times N
$$

scores.

---

Ví dụ:

```text
N = 8192
```

Attention matrix:

```text
67 triệu phần tử
```

---

Quá trình:

```text
HBM
 │
 ▼

Load Q

HBM
 │
 ▼

Load K

HBM
 │
 ▼

Store Attention

HBM
 │
 ▼

Read Attention

HBM
 │
 ▼

Store Softmax

HBM
 │
 ▼

Read Softmax

HBM
 │
 ▼

Multiply V
```

---

Rất nhiều truy cập HBM.

---

# 7. Vì sao Flash Attention tập trung vào IO?

Bài báo Flash Attention đưa ra quan sát:

> FLOPs không phải bottleneck chính.

Mà là:

```text
HBM Access
```

---

Mục tiêu:

```text
Giảm đọc HBM

Giảm ghi HBM

Tăng tái sử dụng SRAM
```

---

# 8. Data Movement trong Standard Attention

```text
HBM

Q
K
V

 │
 ▼

QKᵀ

 │
 ▼

Store N×N Matrix

 │
 ▼

Read N×N Matrix

 │
 ▼

Softmax

 │
 ▼

Store Softmax

 │
 ▼

Read Softmax

 │
 ▼

Multiply V
```

---

Vấn đề:

```text
Attention Matrix = O(N²)
```

---

# 9. Data Movement trong Flash Attention

Flash Attention chia dữ liệu thành Tile.

---

Minh họa:

```text
HBM

Q Tile
K Tile
V Tile

 │
 ▼

Load vào SRAM

 ┌──────────────┐
 │    SRAM      │
 │  Tile Cache  │
 └──────────────┘

 │
 ▼

Tensor Core

 │
 ▼

Online Softmax

 │
 ▼

Output Tile
```

---

Không tạo:

```text
N × N
Attention Matrix
```

---

Dữ liệu được:

```text
Load một lần

↓

Tái sử dụng nhiều lần

↓

Ghi kết quả cuối cùng
```

---

## So sánh

```text
Standard Attention

HBM ↔ HBM ↔ HBM ↔ HBM

Rất nhiều IO


Flash Attention

HBM
 ↓

SRAM

 ↓

Tensor Core

 ↓

Output

Ít IO hơn nhiều
```

---

# 10. Tổng kết

Flash Attention không thay đổi:

$$
Softmax(QK^T)V
$$

về mặt toán học.

---

Đóng góp lớn nhất là:

```text
IO-Aware Optimization
```

---

Tư tưởng cốt lõi:

```text
Đừng tối ưu FLOPs trước.

Hãy tối ưu Data Movement trước.
```

---

Flash Attention đạt được điều này bằng cách:

1. Chia Attention thành Tile.
2. Load Tile vào SRAM.
3. Tái sử dụng dữ liệu trong SRAM.
4. Giảm truy cập HBM.
5. Không materialize Attention Matrix.
6. Sử dụng Online Softmax.

---

Tóm tắt:

```text
HBM
 = Dung lượng lớn
 = Chậm hơn

SRAM
 = Dung lượng nhỏ
 = Rất nhanh

Flash Attention
 = Tối đa SRAM
 = Tối thiểu HBM
 = Tăng Throughput
 = Giảm Memory
```

---
