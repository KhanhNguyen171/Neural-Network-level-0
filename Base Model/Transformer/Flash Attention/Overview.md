## Flash Attention Architecture Overview

```text
┌───────────────────────────────────────────────────────────────────────┐
│                         FLASH ATTENTION                              │
│      Fast and Memory-Efficient Exact Attention for Transformers      │
└───────────────────────────────────────────────────────────────────────┘


                    Standard Attention

       Q                    K                    V
       │                    │                    │
       ▼                    ▼                    ▼

               ┌───────────────────────┐
               │       Q × Kᵀ          │
               │   N × N Matrix        │
               └──────────┬────────────┘
                          │
                          ▼

                     Softmax

                          │
                          ▼

                       P × V

                          │
                          ▼

                       Output


Problems
────────

✗ Materialize N×N Attention Matrix
✗ O(N²) Memory
✗ Large HBM Traffic
✗ Memory Bottleneck
```

```text
┌───────────────────────────────────────────────────────────────────────┐
│                       FLASH ATTENTION FLOW                           │
└───────────────────────────────────────────────────────────────────────┘


           HBM (Global Memory)
    ┌──────────────────────────────┐
    │                              │
    │        Q     K     V         │
    │                              │
    └──────────────┬───────────────┘
                   │
                   ▼

          Load Small Tiles Only

                   │
                   ▼

    ┌──────────────────────────────┐
    │       SRAM / Shared Memory   │
    │                              │
    │    Qi     Kj     Vj          │
    │                              │
    └──────────────┬───────────────┘
                   │
                   ▼

         Compute Tile Attention

                   │
                   ▼

             Sij = QiKjᵀ

                   │
                   ▼

            Online Softmax

         Running Max (m)

                   │

         Running Sum (l)

                   │
                   ▼

          Partial Output Oij

                   │
                   ▼

          Accumulate Output

                   │
                   ▼

            Final Output O


✓ No N×N Matrix Materialization
✓ Exact Attention
✓ O(N) Memory
✓ Reduced HBM Access
```

---

### Tile-Based Processing

```text
               Full Attention Matrix

             K1      K2      K3

       ┌──────┬──────┬──────┐
   Q1  │██████│██████│██████│
       ├──────┼──────┼──────┤
   Q2  │██████│██████│██████│
       ├──────┼──────┼──────┤
   Q3  │██████│██████│██████│
       └──────┴──────┴──────┘


Each block is processed independently

(Qi,Kj,Vj)

without storing the full matrix.
```

---

### Online Softmax

```text
Tile 1
   │
   ▼

m₁ = max(Tile1)
l₁ = sum(exp)

   │
   ▼

Tile 2

m₂ = max(m₁, Tile2)

l₂ =
exp(m₁-m₂)l₁
+
exp(Tile2-m₂)lTile2

   │
   ▼

Tile 3

update again

   │
   ▼

Exact Global Softmax
```

---

### IO-Aware Design

```text
Traditional Attention


HBM
 │
 ▼

Compute

 │
 ▼

HBM

 │
 ▼

Compute

 │
 ▼

HBM

Many expensive memory accesses



Flash Attention


HBM
 │
 ▼

SRAM
 │
 ▼

Tensor Core
 │
 ▼

SRAM Reuse
 │
 ▼

Output

Minimal memory traffic
```

---

### Forward + Backward

```text
FORWARD

Load Tile
    │
    ▼

Compute Attention

    │
    ▼

Online Softmax

    │
    ▼

Store:

m
l
Output


Do NOT Store:

QKᵀ
Softmax(QKᵀ)



BACKWARD

Reload Tile

    │
    ▼

Recompute Attention

    │
    ▼

Compute Gradients
```

---

### Complexity Comparison

```text
Memory

│
│
│                       Standard Attention
│
│                    /
│                  /
│                /
│              /
│            /
│          /
│        /
│      /
│    /
│  /
│/
├────────────────────────────────────► Sequence Length

 Flash Attention
 ───────────────────────────────

 O(N)

 Standard Attention

 O(N²)
```

---

### Flash Attention Design Philosophy

```text
            FLOPs
              │
              ▼

       NOT THE MAIN
        BOTTLENECK


              │
              ▼

      DATA MOVEMENT

              │
              ▼

        HBM Access

              │
              ▼

     FLASH ATTENTION

              │
              ▼

  Minimize Memory Traffic

              │
              ▼

      Higher Throughput
      Lower Memory Usage
      Longer Context Length
```
