# Source Code Architecture

Thư mục `src/` chứa các thành phần cốt lõi của một Convolutional Neural Network (CNN) được xây dựng từ đầu nhằm mục đích học tập và trực quan hóa thuật toán.

```
src/
│
├── convolution.py
├── pooling.py
├── activation.py
├── flatten.py
├── dense.py
├── simple_cnn.py
└── utils.py
```

Mỗi file tương ứng với một phép biến đổi toán học trong CNN.

---
# Design Philosophy

Project được chia thành hai giai đoạn:

## Phase A — Understanding CNN

Tập trung vào:

- Convolution
- ReLU
- Pooling
- Flatten
- Dense Layer
- Forward Propagation

Mục tiêu: __Hiểu cách CNN xử lý dữ liệu từ ảnh đầu vào đến dự đoán cuối cùng.__

---
## Phase B — Building CNN From Scratch

Tập trung vào:

- Gradient
- Chain Rule
- Backpropagation
- Loss Function
- Training Loop

Mục tiêu: __Xây dựng một mini CNN framework có khả năng học từ dữ liệu.__

---

# CNN Data Flow

```
Input Image
      │
      ▼
Convolution
      │
      ▼
Activation
      │
      ▼
Pooling
      │
      ▼
Flatten
      │
      ▼
Dense Layer
      │
      ▼
Prediction
```

---

# Computational Graph
Forward Propagation:
```
Input
  │
  ▼
Conv
  │
  ▼
ReLU
  │
  ▼
Pool
  │
  ▼
Flatten
  │
  ▼
Dense
  │
  ▼
Prediction
```

Backward Propagation:
```
Loss
  ▲
  │
Dense
  ▲
  │
Flatten
  ▲
  │
Pool
  ▲
  │
ReLU
  ▲
  │
Conv
  ▲
  │
Input
```

---

# convolution.py

## Vai trò

Thực hiện phép tích chập (Convolution), thành phần quan trọng nhất của CNN.

Convolution có nhiệm vụ trích xuất đặc trưng từ ảnh đầu vào như:

* Cạnh (Edges)
* Góc (Corners)
* Đường thẳng (Lines)
* Kết cấu ảnh (Textures)
* Mẫu đặc trưng (Patterns)

---

## Chứa

```
Convolution2D
│
├── Parameters
│   ├── filters
│   └── bias
│
├── Cache
│   └── input
│
├── Gradients
│   ├── d_filters
│   └── d_bias
│
├── compute_output_shape()
├── convolve_single_channel()
├── convolve_multi_channel()
│
├── forward()
└── backward()
```

---
## Forward

$$Y = X * W + b$$

Trong đó:
- $X$: input
- $W$: Filter
- $b$: Bias

## Backward
CNN cần tính:
$$\frac {\partial L} {\partial W}$$
$$\frac {\partial L} {\partial b}$$
$$\frac {\partial L} {\partial X}$$

Để cập nhật filter trong quá trình học.

---

## Input

```
H × W × C
```

Trong đó:

* H: Height
* W: Width
* C: Number of Channels

Ví dụ:

```
28 × 28 × 1
```

hoặc

```
224 × 224 × 3
```

---

## Kernel

```
K × K × C
```

Ví dụ:

```
3 × 3 × 3
```

---

## Output

```
Hout × Wout
```

hoặc

```
F × Hout × Wout
```

với:

- F là số lượng Filters
- Hout = chiều cao đầu ra
- Wout = chiều rộng đầu ra

---

Output Shape Formula
$$H_{out} = \lfloor  \frac {H - K + 2P} {S} \rfloor + 1$$

$$W_{out} = \lfloor  \frac {W - K + 2P} {S} \rfloor + 1$$

phép lấy phần nguyên xuống.

Trong đó:
- $K$: Kernel Size
- $P$: Padding
- $S$: Stride

---

## Công thức

Tại mỗi vị trí:

```
output(i,j)
=
Σ input × kernel
```

Hay:
$$Output(i, j) = \sum_c \sum_m \sum_n Input_{c, i+m, j+n} \times Kernel_{c, m, n}$$

---

## Vai trò trong CNN

```
Input Image
      │
      ▼
Convolution
      │
      ▼
Feature Maps
```

---

# activation.py

## Vai trò

Đưa tính phi tuyến (Non-Linearity) vào mạng nơ-ron.

Nếu không có Activation Function, toàn bộ CNN chỉ là một phép biến đổi tuyến tính lớn.

---

## Chứa

```
Activation
│
├── ReLU
├── Sigmoid
├── Tanh
└── Softmax
```

---

## ReLU
Forward:
$$ReLU(x) = max(0, x)$$

backward:
$$ReLU'(x) = \begin{cases}
1& x > 0\\ 
0& x \le 0
\end{cases}$$

ReLU là hàm kích hoạt được sử dụng phổ biến nhất trong CNN.

Ví dụ:

```
Input

[-5, -2, 3, 7]

Output

[0, 0, 3, 7]
```

---

## Sigmoid
Forward:
$$\sigma (x) = \frac {1} {1 + e^{-x}}$$

Backward:
$$\sigma'(x) = \sigma (x) (1 - \sigma(x))$$

## Softmax
Forward:
$$P_i = \frac {e^{z_i}} {\sum_j e^{z_j}}$$

Backward:

Softmax có Jacobian matrix:
$$
\frac{\partial y_i}{\partial z_j}
=
y_i(\delta_{ij}-y_j)
$$

Trong thực tế Deep Learning, Softmax thường được kết hợp với Cross Entropy để đơn giản hóa gradient.


tổng xác suất:
$$\sum_i P_i = 1$$

---

# pooling.py

## Vai trò

Giảm kích thước Feature Map nhưng vẫn giữ lại thông tin quan trọng.

Mục tiêu:

* Giảm số lượng tham số
* Giảm chi phí tính toán
* Giảm Overfitting

---

## Chứa

```
Pooling
│
├── MaxPooling2D
├── AveragePooling2D
├── backward()
└── forward()
```

---

## Max Pooling

Ví dụ:

```
1 4
3 2
```

↓

```
4
```

---

## Công thức

$$Output = max(Window)$$

Forward:
$$y = max(x)$$
Backward: __Gradient chỉ được truyền tới phần tử lớn nhất trong Pooling Window.__

Ví dụ:
```
1 5
2 3
```

Forward: `5`

Backward:
```
0 1
0 0
```

### Average Pooling

$$Output = \frac {1} {N} \sum^N_{i=1} x_i$$

Backward:
$$\frac {\partial L} {\partial x_i} = \frac {1} {N} \frac {\partial L} {\partial y}$$
---

## Vai trò trong CNN

```
Feature Map
      │
      ▼
Pooling
      │
      ▼
Smaller Feature Map
```

---

# flatten.py

Forward:
$$(C, H, W) \rightarrow (N)$$

Backward:
$$(N) \rightarrow (C, H, W)$$

## Vai trò

Chuyển Feature Map nhiều chiều thành Vector 1 chiều.

---

## Chứa

```
Flatten
│
├── backward()
└── forward()
```

---

## Ví dụ

Input:

```
16 × 8 × 8
```

↓

Output:

```
1024
```

vì:

$$16 \times 8 \times 8 = 1024$$

---

## Vai trò trong CNN

```
Feature Maps
      │
      ▼
Flatten
      │
      ▼
Feature Vector
```

---

# dense.py

## Vai trò

Mô phỏng tầng Fully Connected Layer trong mạng nơ-ron truyền thống.

---

## Luồng xử lý

```
Flatten
   │
   ▼
Input Vector x
   │
   ▼
z = Wx + b
   │
   ▼
Activation
   │
   ▼
Output
```

---

## Chứa

```
Dense
│
├── Parameters
│   ├── weights
│   └── bias
│
├── Cache
│   └── input
│
├── Gradients
│   ├── dW
│   └── db
│
├── forward()
└── backward()
```

---

## Công thức
### Forward:
$$z = Wx + b$$

Trong đó:

* x: Input Vector
* W: Weight Matrix
* b: Bias Vector
* z: Linear Output

### Backward:
Gradient của Weight:
$$\frac {\partial L} {\partial W} = \frac {\partial L} { \partial z} x^T$$

Gradient của Bias:
$$\frac {\partial L} {\partial b} = \sum \frac {\partial L} { \partial z}$$

Gradient truyền ngược:
$$\frac {\partial L} {\partial x} = W^T \frac {\partial L} { \partial z}$$

---

## Ví dụ

$$
x=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
$$

$$
W=
\begin{bmatrix}
0.1 & 0.2 & 0.3\\
0.4 & 0.5 & 0.6
\end{bmatrix}
$$

$$
b=
\begin{bmatrix}
0.1\\
0.2
\end{bmatrix}
$$

Tính:
$$
Wx =
\begin{bmatrix}
1.4\\
3.2
\end{bmatrix}
$$


Kết quả:

$$
z=
\begin{bmatrix}
1.5\\
3.4
\end{bmatrix}
$$

---

## Vai trò trong CNN

```
Flatten
     │
     ▼
Dense Layer
     │
     ▼
Class Scores
```

---

# simple_cnn.py

## Vai trò

Ghép toàn bộ các thành phần thành một CNN hoàn chỉnh.

---

## Kiến trúc

```
Image
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Flatten
 ↓
Dense
 ↓
Softmax
```

## Training
```
Forward
   │
   ▼
Loss
   │
   ▼
Backward
   │
   ▼
Gradients
   │
   ▼
Update Weights
```

### SGD:
$$W = W - \eta \frac {\partial L} {\partial W}$$

$$b = b - \eta \frac {\partial L} {\partial b}$$

- $W$: Weight
- $\eta$: Learning Rate

---
# Loss Function

## Mean Squared Error

$$
L
=
\frac1N
\sum_i
(y_i-\hat y_i)^2
$$

## Cross Entropy

$$
L
=
-\sum_i
y_i\log(\hat y_i)
$$
---
# Chain Rule

Backpropagation dựa trên Chain Rule:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial x}
$$

Ví dụ:

$$
L=f(g(x))
$$

$$
\frac{dL}{dx}
=
\frac{dL}{dg}
\cdot
\frac{dg}{dx}
$$
---

# utils.py

## Vai trò

Chứa các hàm hỗ trợ trực quan hóa và xử lý dữ liệu.

Ví dụ:

```
load_image()
show_image()
plot_feature_map()
normalize_image()
```

Các hàm này không thuộc thuật toán CNN nhưng giúp quan sát dữ liệu dễ dàng hơn trong notebook.
---

# Layer Design
```
Layer
│
├── Parameters
├── Cache
├── Gradients
│
├── forward()
└── backward()
```
## Parameters
Các giá trị được học

VD:
```
weights
bias
filters
```

## Cache
Lưu dữ liệu cho Backpropagation.

Ví dụ:
```
Lưu dữ liệu cho Backpropagation.
```

## Gradients
Lưu gradient được tính trong Backward.

Ví dụ:
```
Lưu gradient được tính trong Backward.
```

---
# CNN Learning Pipeline
```
Image
  │
  ▼
Forward
  │
  ▼
Prediction
  │
  ▼
Loss
  │
  ▼
Backward
  │
  ▼
Gradients
  │
  ▼
Weight Update
  │
  ▼
Repeat
```

---
# Mục tiêu học tập

Sau khi hoàn thành toàn bộ project, người học sẽ hiểu:

- Ảnh được biểu diễn như thế nào trong CNN.
- Convolution tạo Feature Maps ra sao.
- ReLU tạo tính phi tuyến như thế nào.
- Pooling giảm chiều dữ liệu như thế nào.
- Dense Layer thực hiện phân loại ra sao.
- Gradient được tính như thế nào.
- Backpropagation lan truyền lỗi như thế nào.
- Weight được cập nhật như thế nào.
- Một mini Deep Learning Framework được xây dựng từ đầu ra sao.
- PyTorch và TensorFlow đang tự động hóa những bước nào trong toàn bộ quy trình huấn luyện CNN.