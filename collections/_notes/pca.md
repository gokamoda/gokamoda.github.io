---
layout: splash
title: "メモ: PCA"
use_math: true
header:
  teaser: /assets/img/pca.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2025-08-06 # moved from DissectingGPT
excerpt: "PCAの基礎と実装"
---

$n$ 個の $d$ 次元のデータ $\bm{X}$ を、$d'$ 次元に圧縮するための線形変換行列 $\bm{W}$ を求める。

$$

\bm{X}
=\begin{bmatrix}
\bm{x}_1 \\
\bm{x}_2 \\
\vdots \\
\bm{x}_n
\end{bmatrix}
= \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n,1} & x_{n,2} & \cdots & x_{n,d}
\end{bmatrix}
\in \mathbb{R}^{n \times d}
$$

$$
\bm{W}
= \begin{bmatrix}
\bm{w}_1 & \bm{w}_2 & \cdots & \bm{w}_{d'} \\
\end{bmatrix}
= \begin{bmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,d'} \\
w_{2,1} & w_{2,2} & \cdots & w_{2,d'} \\
\vdots & \vdots & \ddots & \vdots \\
w_{d,1} & w_{d,2} & \cdots & w_{d,d'}
\end{bmatrix}
\in \mathbb{R}^{d \times d'}
$$

変換後のデータ $\bm{Y}$ は次のように表される。
$$
\bm{Y} = \bm{X} \bm{W} \in \mathbb{R}^{n \times d'}
$$

PCAでは、圧縮後の各次元の分散を最大化することを考える。  
圧縮後の $j\ (1\le j \le d')$次元目 の分散は、定義より以下で表せる:

$$
\begin{align}
s_j^2 
&= \frac{1}{n}\sum_{i=1}^{n}(y_{ij}-\bar{y_j})^2\\
&= \frac{1}{n}\sum_{i=1}^{n}\left(
    \bm{x}_i \bm{w}_j - \frac{1}{n}\sum_{k=1}^{n}\bm{x}_k \bm{w}_j
\right)^2 \\
&= \frac{1}{n}\sum_{i=1}^{n}\left(
    \left(\bm{x}_i - \frac{1}{n}\sum_{k=1}^{n}\bm{x}_k\right) \bm{w}_j
\right)^2 \\
\end{align}
$$

ただし $\bar{y_j} = \frac{1}{n}\sum_{i=1}^{n}y_{ij}$ は $j$ 次元目の平均値を表す。

ここで、$\bar{\bm{x}} = \frac{1}{n}\sum_{i=1}^{n}\bm{x}_i \in \mathbb{R}^{d}$ とし、 中心化 (各次元の平均をゼロに) されたデータを $\bm{X}^c$ とする。 $\bm{X}^c$ は以下で定義される。

$$
\begin{align}
\bm{X}^c
&=
\begin{bmatrix}
\bm{x}_1 \\
\bm{x}_2 \\
\vdots \\
\bm{x}_n
\end{bmatrix}
- 
\begin{bmatrix}
\bar{\bm{x}} \\
\bar{\bm{x}} \\
\vdots \\
\bar{\bm{x}} \\
\end{bmatrix}
=\begin{bmatrix}
\bm{x}_1^c \\
\bm{x}_2^c \\
\vdots \\
\bm{x}_n^c
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\bm{x}^c_i
&=\bm{x}_i - \bar{\bm{x}}\\
&=\bm{x}_i - \frac{1}{n}\sum_{k=1}^{n}\bm{x}_k
\end{align}
$$

すると、次元 $j\ (1\le j \le d')$ の分散は以下で表せる:

$$
\begin{align}
s_j^2 
&= \frac{1}{n}\sum_{i=1}^{n}(\bm{x}_i^c \bm{w}_j)^2 \\
&= \frac{1}{n}\sum_{i=1}^{n}(\bm{x}_i\bm{w}_j^c)^\top (\bm{x}_i^c \bm{w}_j) \\
&= \frac{1}{n}\sum_{i=1}^{n}\bm{w}_j^\top(\bm{x}_i^c\top\bm{x}_i^c)\bm{w}_j \\
&= \frac{1}{n}\bm{w}_j^\top \left(\sum_{i=1}^{n}\bm{x}_i^c\top\bm{x}_i^c\right) \bm{w}_j \\
&= \bm{w}_j^\top \left(\frac{1}{n}\bm{X}^{c\top}\bm{X}^c\right) \bm{w}_j \\
&= \bm{w}_j^\top \bm{S} \bm{w}_j
\end{align}
$$

ここで、$\bm{S} \in \mathbb{R}^{d \times d}$ はデータ$\bm{X}$の、共分散行列と呼ばれる。

PCAでは、$s_j^2 (1\le j \le d')$を最大化する $\bm{w}_j$ を求める。
なお、ここで$\bm{w}_j$を大きくすれば$s_j^2$も大きくなるため、$\bm{w}_j$は単位ベクトル、つまり $$ \|\bm{w}_j\|^2 = \bm{w}_j^\top \bm{w}_j = 1 $$ の成約を設ける。


結果、解きたい問題は

$$
\underset{\bm{w}_j}{\text{argmax}}\quad \bm{w}_j^\top \bm{S} \bm{w}_j \quad \text{subject to}\quad  \bm{w}_j^\top \bm{w}_j = 1
$$

これをラグランジュの未定乗数法を用いて解くことを考える。ラグランジュ関数は以下のようになる。

$$
\begin{align}
\mathcal{L}(\bm{w}_j, \lambda) &= \bm{w}_j^\top \bm{S} \bm{w}_j - \lambda (\bm{w}_j^\top \bm{w}_j - 1) \\
\end{align}
$$

これを$W_j$で微分すれば、

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \bm{w}_j} &= 2 \bm{S} \bm{w}_j - 2 \lambda \bm{w}_j
\end{align}
$$

したがって、以下を満たすときに極値をとる。

$$
\bm{S} \bm{w}_j = \lambda \bm{w}_j
$$

したがって、$\bm{w}_j$は$\bm{S}$の固有ベクトルであり、$\lambda$は対応する固有値である。  
また、$s_j^2$は、以下のように書き換えられる。
$$
\begin{align}
s_j^2 &= \bm{w}_j^\top \bm{S} \bm{w}_j \\
&= \bm{w}_j^\top \lambda \bm{w}_j \\
&= \lambda \bm{w}_j^\top \bm{w}_j \\
&= \lambda
\end{align}
$$

PCAでは、$d'$ 個の次元の分散を最大化することを考えるため、$\bm{S}$の固有値が大きい順に対応する$d'$ 個の固有ベクトルをとり並べたものを$\bm{W}$とすればよい。


## (復習) 固有値分解


行列 $\bm{A} \in \mathbb{R}^{d\times d}$ の固有値分解は、直交行列$\bm{V}$を用いて以下のように表される。


$$
\bm{A} = \bm{V} \bm{\Lambda} \bm{V}^\top
$$

ここで、

$$
\bm{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)
$$

$$
\bm{V} = \begin{bmatrix}
\bm{v}_1 & \bm{v}_2 & \cdots & \bm{v}_d
\end{bmatrix}
$$

とする。 $\lambda_i$ は $\bm{A}$ の固有値、$\bm{v}_i$ は対応する固有ベクトルである。

また、以下のように変形できる。

$$
\begin{align}
\bm{A} &= \bm{V} \bm{\Lambda} \bm{V}^\top\\
&= 
\begin{bmatrix}
\bm{v}_1 & \bm{v}_2 & \cdots & \bm{v}_d
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \bm{v}_1 \\
\lambda_2 \bm{v}_2 \\
\vdots \\
\lambda_d \bm{v}_d
\end{bmatrix} \\
&= \sum_{i=1}^{d} \lambda_i \bm{v}_i \bm{v}_i^\top
\end{align}
$$




## $\bm{S}$の固有値/ベクトルは$X$の特異値分解 (SVD) で求められる

再掲: $\bm{S} \in \mathbb{R}^{d\times d}$, $\bm{X}^c \in \mathbb{R}^{n\times d}$.


変換行列 $\bm{W} \in \mathbb{R}^{d\times d'}$ を求めるために、$\bm{S}$の固有値を求める必要がある。
ここで、$\bm{S}$ を $\bm{X}^c$ に関して書き換え、$\bm{X}^c$ の特異値分解 (SVD) を用いると以下のようになる。


$$
\begin{align}
\bm{S} &= \frac{1}{n} \bm{X}^{c\top} \bm{X}^c \\
&= \frac{1}{n} \left(\bm{U}\bm{\Sigma} \bm{V}^{\top}\right)^\top \left(\bm{U}\bm{\Sigma} \bm{V}^{\top}\right) \\
&= \frac{1}{n} \bm{V} \bm{\Sigma}^\top \bm{U}^\top \bm{U} \bm{\Sigma} \bm{V}^{\top} \\
&= \frac{1}{n} \bm{V} \bm{\Sigma}^{2} \bm{V}^{\top}\quad (\because \bm{U} \text{ is orthogonal})\\
&= \bm{V} \left(\frac{\bm{\Sigma}^{2}}{n} \right) \bm{V}^{\top} \\
\end{align}
$$


つまり、以下が言える:
- $\bm{S}$ の 固有値は [{$\bm{X}^c$ を特異値分解して得られる固有値からなる行列$\bm{\Sigma}$} を用いて表される $\bm{\Sigma}^{2}/n$ ]の各対角成分
- $\bm{S}$ の 固有ベクトルは $\bm{X}^c$ を特異値分解して得られる $\bm{V}$ の各列ベクトルである。

したがって、PCAを行う際は、$\bm{X}^{c\top}\bm{X}^c$ を計算する必要はなく、$\bm{X}^c$ を特異値分解して、$\bm{V}$ のうち、対応する固有値が大きい $d'$ 個の列ベクトルをとり並べたものを $\bm{W}$ とすればよい。


## 重み付きPCA

PCAでは、各データ点のサンプリング確率が等しいことを仮定している (もちろん、その分重複を許してサンプリングすれば良い)。
各点のサンプリング確率 $p(x)$ を考慮した重み付きのPCAについて考える。

まず、中心化について、各点のサンプリング確率を考慮した平均 $\bar{\bm{x}}$ は以下で定義される:

$$
\bar{\bm{x}} := \sum_{i} p(x_i) \bm{x}_i
$$

この平均を使って中心化されたデータを再度 $\bm{X}^c$ とする。

次に分散について、次元 $j\ (1\le j \le d')$ の分散は以下で表せる:

$$
\begin{align}
s_j^2 
&= \frac{1}{n}\sum_{i=1}^{n}p(x_i)(\bm{x}_i^c \bm{w}_j)^2 \\
&= \frac{1}{n}\sum_{i=1}^{n}p(x_i)(\bm{x}_i\bm{w}_j^c)^\top (\bm{x}_i^c \bm{w}_j) \\
&= \frac{1}{n}\sum_{i=1}^{n}\bm{w}_j^\top(p(x_i)\bm{x}_i^c\top\bm{x}_i^c)\bm{w}_j \\
&= \frac{1}{n}\bm{w}_j^\top \left(\sum_{i=1}^{n}\left(\sqrt{p(x_i)}\bm{x}_i^c\right)^\top\left(\sqrt{p(x_i)}\bm{x}_i^c\right)\right) \bm{w}_j \\
&= \frac{1}{n}\bm{w}_j^\top \left(\bm{P}(X) \bm{X}^c \right)^\top\left(\bm{P}(X) \bm{X}^c \right)\bm{w}_j \\
&= \bm{w}_j^\top \bm{S} \bm{w}_j
\end{align}
$$

ただし $\bm{P}(X) = \diag{\sqrt{p(x_1)}, \ldots, \sqrt{p(x_n)}}$ とする。

したがって、通常のPCAと同様に考えれば、PCAは $\bm{P}(X) \bm{X}^c $ の右特異ベクトルを並べれば良いことがわかる