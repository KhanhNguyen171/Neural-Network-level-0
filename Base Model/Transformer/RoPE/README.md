# Rotary Position Embedding (RoPE)

## 1. Giới thiệu

Rotary Position Embedding (RoPE) là phương pháp mã hóa vị trí được đề xuất trong bài báo:

**RoFormer: Enhanced Transformer with Rotary Position Embedding**

Mục tiêu của RoPE là đưa thông tin vị trí vào Attention mà không cần cộng thêm vector positional embedding vào token embedding như Transformer gốc.

Ý tưởng cốt lõi:

* Position không được cộng vào embedding.
* Position được mã hóa bằng phép quay (rotation) trong không gian vector.
* Attention tự động học được khoảng cách tương đối giữa các token.

RoPE hiện là nền tảng của phần lớn LLM hiện đại:

* GPT-NeoX
* LLaMA
* Mistral
* Mixtral
* Qwen
* DeepSeek
* Gemma
* x-transformers

---

# 2. Bài toán của Positional Embedding truyền thống

Transformer không có khái niệm thứ tự.

Attention chỉ nhìn thấy tập token:

$$
X = [x_1,x_2,...,x_n]
$$

Nếu hoán đổi thứ tự:

$$
[x_1,x_2,x_3]
$$

thành

$$
[x_3,x_2,x_1]
$$

thì Self-Attention nguyên bản không phân biệt được.

Do đó cần bổ sung thông tin vị trí.

Transformer gốc sử dụng:

$$
h_i = x_i + p_i
$$

với:

$$
p_i
$$

là positional embedding.

Nhược điểm:

* Position tuyệt đối.
* Khó extrapolate tới sequence dài hơn.
* Attention không mô hình hóa trực tiếp relative distance.

RoPE được thiết kế để giải quyết các vấn đề này.

---

# 3. Ý tưởng hình học

<img src="assets/RoPE_visualize.png">

Xét vector 2 chiều:

$$
v=
\begin{bmatrix}
x \
y
\end{bmatrix}
$$

Phép quay góc $\theta$:

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Sau khi quay:

$$
v' = R(\theta)v
$$

Ta nhận được vector mới nhưng vẫn giữ nguyên độ dài:

$$
||v'|| = ||v||
$$

RoPE sử dụng chính phép quay này để mã hóa vị trí.

---

# 4. Rotary Embedding

Giả sử:

$$
q_i
$$

là query tại vị trí:

$$
i
$$

Ta quay vector theo góc phụ thuộc vị trí:

$$
\widetilde q_i =

R(i\theta)
q_i
$$

Tương tự:

$$
\widetilde k_j =

R(j\theta)
k_j
$$

Attention sử dụng:

$$
\widetilde q_i^T
\widetilde k_j
$$

thay vì:

$$
q_i^T k_j
$$

---

# 5. Tính chất quan trọng nhất

Attention sau khi áp dụng RoPE:

$$
\widetilde{q}_i^T \widetilde{k}_j
=
\left(R(i\theta)q_i\right)^T
\left(R(j\theta)k_j\right)
$$

Khai triển:

$$
=
q_i^T
R(i\theta)^T
R(j\theta)
k_j
$$

Do ma trận quay là ma trận trực giao:

$$
R(\theta)^T
=
R(-\theta)
$$

nên:

$$
=
q_i^T
R(-i\theta)
R(j\theta)
k_j
$$

Sử dụng tính chất:

$$
R(a)R(b)
=
R(a+b)
$$

ta được:

$$
=
q_i^T
R((j-i)\theta)
k_j
$$

Do đó:

$$
\widetilde{q}_i^T \widetilde{k}_j
=
q_i^T
R((j-i)\theta)
k_j
$$

Điều quan trọng nhất là Attention chỉ phụ thuộc vào:

$$
j-i
$$

thay vì phụ thuộc trực tiếp vào vị trí tuyệt đối.

Đây chính là lý do RoPE tự nhiên sinh ra **Relative Position Encoding**.

---

# 6. Mở rộng lên không gian nhiều chiều

Trong Transformer:

$$
d_{model}
\in
\{1024,2048,4096,\dots\}
$$

Không thể áp dụng một phép quay 2 chiều duy nhất cho toàn bộ vector.

RoPE chia vector thành các cặp chiều:

$$
[x_1,x_2,x_3,x_4,\dots,x_d]
$$

thành:

$$
(x_1,x_2),
(x_3,x_4),
\dots,
(x_{d-1},x_d)
$$

Mỗi cặp được xem như một mặt phẳng 2 chiều độc lập.

Ví dụ:

$$
[x_1,x_2,x_3,x_4]
$$

được chia thành:

$$
(x_1,x_2)
$$

và

$$
(x_3,x_4)
$$

Mỗi cặp sẽ được quay bằng một góc khác nhau.

---

# 7. Ma trận RoPE đầy đủ



Cho:

$$
d = head\_dim
$$

Tập tần số:

$$
\Theta
=
\{
\theta_0,
\theta_1,
\dots,
\theta_{\frac d2-1}
\}
$$

với:

$$
\theta_i
=
10000^{-\frac{2i}{d}}
$$

Tại vị trí \(m\), ma trận quay được viết dưới dạng block-diagonal:

$$
R_m
=
\begin{bmatrix}

\cos(m\theta_0)
&
-\sin(m\theta_0)
&
0
&
0
&
\cdots
\\

\sin(m\theta_0)
&
\cos(m\theta_0)
&
0
&
0
&
\cdots
\\

0
&
0
&
\cos(m\theta_1)
&
-\sin(m\theta_1)
&
\cdots
\\

0
&
0
&
\sin(m\theta_1)
&
\cos(m\theta_1)
&
\cdots
\\

\vdots
&
\vdots
&
\vdots
&
\vdots
&
\ddots

\end{bmatrix}
$$

Mỗi cặp chiều được quay với một tần số riêng.

Những chiều đầu có tần số cao.

Những chiều cuối có tần số thấp.

Điều này giúp mô hình đồng thời học được:

- Quan hệ cục bộ
- Quan hệ dài hạn

---

## 7.1. Các chế độ sắp xếp dữ liệu

### Interleave Mode
<img src="assets/iterleave_model.png">

Giải thích kiến trúc: Chế độ này gom cụm xử lý theo từng cặp phần tử liền kề chẵn-lẻ trong không gian bộ nhớ contiguous.

- Vector $\mathbf{x}_{\text{new}}$ được tạo ra bằng cách hoán vị cục bộ: đảo vị trí phần tử lẻ lên trước, phần tử chẵn ra sau và đổi dấu phần tử đứng trước: $[-x_1, x_0, -x_3, x_2, \dots]$.

- Thao tác tính toán diễn ra song song trực tiếp trên thanh ghi vùng nhớ cục bộ lân cận, tối ưu cache cho các phép toán phần tử (Element-wise).


### So sánh Interleave vs Half
<img src="assets/sosanh_iterleave_half.png">

Giải thích kiến trúc: Sơ đồ so sánh hai layout sắp xếp dữ liệu phổ biến nhất trong các mô hình sinh thế hệ mới:

- __Interleave Mode (trái)__: Như định nghĩa trên, bắt cặp đan xen $[x_0, x_1]$. Cấu trúc này được áp dụng mặc định trong các LLM hàng đầu như LLaMA, DeepSeek, Qwen.

- __Half Mode (phải)__: Thay vì bắt cặp liền kề, hệ thống cắt đôi vector theo chiều dài kích thước $d$. Nửa đầu $[x_0, x_1, \dots]$ sẽ bắt cặp tương ứng với nửa sau $[x_{d/2}, x_{d/2+1}, \dots]$. Vector $\mathbf{x}_{\text{new}}$ tạo ra bằng cấu trúc $[-x_{half2}, x_{half1}]$. Kiến trúc này xuất hiện nhiều trong các mô hình khuếch tán không gian thời gian lớn như OpenSORA, StepVideo, hoặc khi xử lý các cấu trúc Tensor dạng hình khối (Cube Rotate).


### Quarter Mode
<img src="assets/quarterModel.png">

Giải thích kiến trúc: Minh họa các layout biến thể phức tạp hơn được sinh ra để tương thích với cấu trúc dữ liệu đa chiều đa phân đoạn:

- __Quarter Mode__: Vector đặc trưng được chia thành 4 phân đoạn bằng nhau. Các phép toán xoay hoán vị được thực hiện chéo giữa phân đoạn 1-2 và phân đoạn 3-4 hoặc xoay vòng tròn.

- __Interleave-Half Mode__: Kiến trúc lai (Hybrid Layout). Hệ thống áp dụng cấu trúc chia đôi (Half) trên tổng thể nhưng bên trong mỗi nửa lại thực hiện cấu trúc đan xen (Interleave). Các layout này tối ưu hóa luồng nạp dữ liệu cho các phần cứng NPU chuyên dụng nhằm tối đa hóa hiệu suất phần trăm sử dụng toán tử SIMD.

---

# 8. Biểu diễn bằng số phức

Đây là cách diễn giải toán học đẹp nhất của RoPE.

Xét một cặp chiều:

$$
(x,y)
$$

Ta định nghĩa:

$$
z=x+iy
$$

với:

$$
i^2=-1
$$

Một phép quay góc \(\theta\):

$$
z'
=
ze^{i\theta}
$$

Theo công thức Euler:

$$
e^{i\theta}
=
\cos\theta
+
i\sin\theta
$$

Do đó:

$$
z'
=
z
(
\cos\theta
+
i\sin\theta
)
$$

chính là phép quay trong mặt phẳng 2 chiều.

RoPE áp dụng ý tưởng này cho Query và Key:

$$
q_m
=
q
e^{im\theta}
$$

$$
k_n
=
k
e^{in\theta}
$$

Attention:

$$
q_m
\overline{k_n}
=
q
\overline{k}
e^{i(m-n)\theta}
$$

với:

$$
\overline{k}
$$

là liên hợp phức.

Ta thấy ngay:

$$
m-n
$$

xuất hiện tự nhiên.

Điều đó có nghĩa:

$$
Attention
\propto
(m-n)
$$

chứ không phải:

$$
m
\quad \text{hoặc} \quad
n
$$

Đây chính là bản chất toán học của Relative Position Encoding trong RoPE.

---

# 9. Thuật toán triển khai

## Bước 1

Sinh vector tần số:

$$
\theta_i
=
10000^{-\frac{2i}{d}}
$$

---

## Bước 2

Cho vị trí token:

$$
m
\in
\{0,1,2,\dots\}
$$

Tính:

$$
m\theta_i
$$

---

## Bước 3

Tính:

$$
\cos(m\theta_i)
$$

và

$$
\sin(m\theta_i)
$$

---

## Bước 4

Áp dụng phép quay cho Query.

Nếu:

$$
q=
[q_1,q_2]
$$

thì:

$$
q'
=
\begin{bmatrix}
\cos\phi & -\sin\phi \\
\sin\phi & \cos\phi
\end{bmatrix}
q
$$

với:

$$
\phi=m\theta_i
$$

---

## Bước 5

Áp dụng phép quay tương tự cho Key.

$$
K
\rightarrow
K_{rope}
$$

---

## Bước 6

Tính Attention:

$$
Attention
=
Softmax
\left(
\frac{
Q_{rope}
K_{rope}^{T}
}
{\sqrt d}
\right)
V
$$

---

## 9.1. Bài toán hiệu năng (RoPE-3D)
### Quy trình triển khai
<img src="assets/trien_khai_Rope_3d.png">

__Giải thích kiến trúc:__ Sơ đồ thể hiện luồng xử lý mã hóa vị trí ba chiều (3D Positional Encoding) cho mô hình Transformer xử lý video. Không gian biểu diễn đặc trưng gồm ba thành phần độc lập: chiều cao ($h$), chiều rộng ($w$), và trục thời gian ($t$).

- Cấu trúc Tensor đặc trưng $D$ ban đầu bắt buộc phải đi qua toán tử tách lát (`Slice`) để chia thành các phân đoạn tương ứng với từng trục không gian.

- Tiếp theo, hệ thống thực hiện đồng thời các phép toán nhân (`Mul`) với các hệ số lượng giác $\sin/\cos$ của từng trục và cộng tích lũy (`Add`) để nhúng vị trí. Cuối cùng, luồng dữ liệu được gộp lại bằng toán tử `Concat`.

### Hạn chế cố hữu
<img src="assets/phan_tich_Rope.png">

__Giải thích kiến trúc:__ Sơ đồ phân tích sâu vào thắt nút cổ chai hiệu năng (Performance Bottleneck) của RoPE-3D truyền thống:

- Do đặc thù dữ liệu video, trục thời gian hoặc các kích thước biên thường nhỏ tạo ra các mảnh dữ liệu rất mỏng (Mảnh cắt đuôi - `tail`).

- Việc liên tục gọi lệnh `Slice` và `Concat` trên bộ nhớ ép buộc GPU phải liên tục cấp phát các vùng nhớ đệm trung gian (Intermediate VRAM overhead) và kích hoạt các thao tác dịch chuyển dữ liệu qua lại không cần thiết giữa các tầng bộ nhớ (Memory-bound). Điều này làm sụt giảm nghiêm trọng tốc độ tính toán thực tế của phần cứng, bất chấp việc phép toán bản chất toán học rất đơn giản.

---

# 10. Vì sao RoPE hoạt động tốt

## Relative Position

Attention phụ thuộc:

$$
j-i
$$

thay vì:

$$
i
\quad \text{hoặc} \quad
j
$$

---

## Translation Invariance

Nếu toàn bộ chuỗi được dịch đi:

$$
i
\rightarrow
i+c
$$

$$
j
\rightarrow
j+c
$$

thì:

$$
(j+c)-(i+c)
=
j-i
$$

không thay đổi.

Do đó mô hình bảo toàn khoảng cách tương đối.

---

## Không tăng số tham số

RoPE không có tham số học được.

Không cần embedding table.

Không cần relative bias matrix.

---

## Khả năng extrapolation

Do dựa trên hàm tuần hoàn:

$$
\sin
$$

và

$$
\cos
$$

nên mô hình thường mở rộng context tốt hơn Positional Embedding học được.

---

## Tương thích hoàn toàn với Self-Attention

Không cần thay đổi:

- Multi-Head Attention
- Feed Forward Network
- Residual Connection
- LayerNorm

Chỉ cần thay đổi:

$$
Q
$$

và

$$
K
$$

trước khi Attention được tính toán.

---

# 11. RoPE trong LLM hiện đại

<img src="assets/img_RoPE.png">

__Giải thích kiến trúc:__ Sơ đồ khối chuẩn hóa quy trình xử lý dữ liệu (Data Pipeline) bên trong một Layer Decoder chuẩn của kiến trúc LLM (như LLaMA).

- Từ Tensor đầu vào (X), hệ thống nhân với các trọng số để chiếu tạo ra 3 ma trận $Q, K, V$.

- Lớp xử lý RoPE được đặt trực tiếp như một bộ lọc tiền xử lý ngay trước khi đưa vào lõi Attention tính toán tích vô hướng. Lớp này nhận thông tin mã hóa vị trí hiện tại ($pos$) và thực hiện xoay cấu trúc không gian trên cặp hai ma trận $Q$ và $K$. Riêng ma trận giá trị $V$ hoàn toàn được giữ nguyên không tác động vị trí, đi thẳng vào làm toán tử tích chéo cuối cùng.


Trong Decoder Transformer:

$$
X
\rightarrow
Q,K,V
$$

RoPE chỉ áp dụng cho:

$$
Q
$$

và

$$
K
$$

Không áp dụng cho:

$$
V
$$

Pipeline:

$$
Input
\rightarrow
Embedding
\rightarrow
Q,K,V
\rightarrow
RoPE(Q,K)
\rightarrow
Attention
\rightarrow
MLP
\rightarrow
Output
$$

Hầu hết các LLM hiện đại đều sử dụng kiến trúc này.

---

# 12. Mối liên hệ với Fourier Features

RoPE thực chất là một dạng Fourier Encoding.

Mỗi chiều biểu diễn một tần số:

$$
\omega_i
=
10000^{-\frac{2i}{d}}
$$

Thông tin vị trí được mã hóa bằng:

$$
\sin(\omega_i t)
$$

và

$$
\cos(\omega_i t)
$$

Do đó RoPE có thể được xem là:

$$
\text{Relative Positional Fourier Encoding}
$$

được tích hợp trực tiếp vào cơ chế Attention.

---

# 13. Những cải tiến từ RoPE

## NTK-Aware RoPE

Điều chỉnh tần số nhằm mở rộng context window.

---

## Linear Scaling

Scale vị trí:

$$
m
\rightarrow
\frac{m}{s}
$$

với:

$$
s > 1
$$

---

## Dynamic RoPE

Tần số thay đổi theo độ dài ngữ cảnh.

---

## YaRN

Context có thể đạt hàng trăm nghìn token.

---

## LongRoPE

Mở rộng tới hàng triệu token.

---

## 13.1. Hợp nhất toán tử (Operator Fusion)
<img src="assets/operatorFusion.png">

Giải thích kiến trúc: Minh họa giải pháp tối ưu hóa mức độ biên dịch mã nguồn phần cứng (Compiler level optimization):

- __Kiến trúc cũ (trái)__: Đồ thị tính toán gồm các nút lệnh rời rạc. Dữ liệu đầu ra của phép nhân `Mul` bắt buộc phải ghi xuống bộ nhớ VRAM, sau đó nút lệnh `Add` lại đọc ngược dữ liệu đó lên để tính toán tiếp.

- __Kiến trúc tối ưu (phải)__: Sử dụng kỹ thuật gộp toán tử thành một Kernel đơn nhất dạng `Mul_Mul_Add Fusion`. Toàn bộ chuỗi phép nhân lượng giác và cộng tích lũy vị trí được thực hiện hoàn toàn bên trong bộ nhớ đệm SRAM (thanh ghi chip xử lý), triệt tiêu hoàn toàn độ trễ IO đọc/ghi phần cứng, giúp tăng tốc độ xử lý lên gấp nhiều lần.

## 13.2. Chuyển đổi sang kiến trúc RoPE-Matrix

<img src="assets/toiuuRope.png">

- __Giải thích kiến trúc:__ Sơ đồ mô tả quy trình chuyển đổi cơ chế tính toán hình học từ dạng cắt tỉa không gian sang đại số tuyến tính ma trận phẳng thuần túy (áp dụng cho biến thể `RoPE-half-3D`). Thay vì sử dụng luồng rẽ nhánh phức tạp, toàn bộ các ma trận hệ số $\cos/\sin$ được ánh xạ trực tiếp thành một cấu trúc ma trận quay khối đồng nhất quy mô lớn. Quá trình nhúng thông tin vị trí giờ đây thu gọn về duy nhất một phép toán nhân ma trận đơn giản (`MatMul`), cực kỳ thân thiện với lõi Tensor Core của GPU.

<img src="assets/tongquathoaRope_matrix.png">

__Giải thích kiến trúc:__ Mô hình hóa trừu tượng mức cao (Abstraction level) minh chứng cho sức mạnh của sự đơn giản hóa kiến trúc:

- __Cấu trúc ban đầu:__ Luồng dữ liệu rối rắm với các kết nối đan chéo đứt đoạn để đồng bộ hóa vị trí 3 chiều trên các phân mảnh dữ liệu.

- __Cấu trúc RoPE-Matrix tổng quát hóa:__ Biến đổi đồ thị trở thành một đường thẳng tuyến tính sạch sẽ thông qua một phép nhân ma trận tích hợp. Cải tiến này giúp loại bỏ toàn bộ các logic kiểm tra điều kiện rẽ nhánh trong mã nguồn điều khiển phần cứng, giúp hệ thống chạy ở xung nhịp tối đa, tạo nền tảng vững chắc cho việc mở rộng cửa sổ ngữ cảnh siêu dài lên tới hàng triệu token của các mô hình như LongRoPE.

---

# 14. Điều cần hiểu trước khi học x-transformers

Để đọc mã nguồn RoPE trong x-transformers nên nắm:

## Linear Algebra

- Vector Space
- Inner Product
- Orthogonal Matrix
- Rotation Matrix

## Fourier Analysis

- Frequency
- Period
- Phase
- Fourier Features

## Attention

- Query
- Key
- Value
- Dot Product

## Complex Numbers

- Euler Formula
- Complex Multiplication
- Complex Rotation

Nếu hiểu các nội dung trên thì RoPE chỉ còn là:

> Mã hóa vị trí bằng phép quay trong không gian vector để Attention tự động phụ thuộc vào khoảng cách tương đối giữa các token.

---

# Tóm tắt

RoPE thay thế Positional Embedding bằng phép quay hình học.

$$
q_i
\rightarrow
R(i)q_i
$$

$$
k_j
\rightarrow
R(j)k_j
$$

Attention trở thành:

$$
q_i^T
R(j-i)
k_j
$$

Do đó Self-Attention tự động học được khoảng cách tương đối:

$$
j-i
$$

mà không cần thêm tham số học.

Đây là nền tảng positional encoding của phần lớn LLM hiện đại như GPT-NeoX, LLaMA, Mistral, Qwen, Gemma và DeepSeek.


