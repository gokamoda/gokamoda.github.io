---
layout: splash
title: "One-hot target と確率分布"
use_math: true
header:
  teaser: /assets/img/one-hot-vs-prob.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-03-16
excerpt: "言語モデルの教師ラベルはone-hotでよいのか、という疑問を少し解消するためのメモ。"
---

## TL;DR

言語モデルの次トークン予測では各位置の教師は one-hot だが、データ全体での負の対数尤度 (NLL) 最小化は、各文脈における経験的な次トークン分布へモデル分布を近づけることとして理解できる。

## Cross Entropy Loss

言語モデルの学習は、通常、コーパス中の各位置における次トークンの負の対数尤度を最小化することとして定式化される。
データセットを文脈と次トークンの組

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{M}
$$

と書くと、負の対数尤度は

$$
\mathcal{L}(\theta)
= -\sum_{i=1}^{M}\log p_\theta(y_i\mid x_i)\label{eq:nll}
$$

である。
(平均 loss として定義する場合は、これを $M$ で割ればよいが、以下の最適化解には影響しない。)

ここで、特定の文脈 $x$ の直後にトークン $y$ が出現する回数を $N(x,y)$ とし、文脈 $x$ の総出現回数を

$$
N(x)=\sum_{y'}N(x,y')
$$

とする。
全体の損失 $\mathcal{L}(\theta)$ を、文脈 $x$ ごとにまとめると、

$$
\begin{align}
\mathcal{L}(\theta)
&= -\sum_x\sum_y N(x,y)\log p_\theta(y\mid x)\\
&= -\sum_x N(x)\sum_y \frac{N(x,y)}{N(x)}\log p_\theta(y\mid x)
\end{align}
$$

と書ける。

さらに、経験的な条件付き分布を

$$
\hat p(y\mid x)=\frac{N(x,y)}{N(x)}
$$

と定義すれば、損失は

$$
\mathcal{L}(\theta)
= \sum_x N(x)
\left[-\sum_y \hat p(y\mid x)\log p_\theta(y\mid x)\right]
$$

と書き換えられる。

角括弧の中身は、経験分布 $\hat p(\cdot\mid x)$ とモデル分布 $p_\theta(\cdot\mid x)$ の cross entropy である。
したがって、「one-hot target に対する負の対数尤度を最小化すること (式$\ref{eq:nll}$)」は、同じ文脈 $x$ ごとに集約して見れば、「経験分布とモデル分布の cross entropy を最小化すること」に対応している。

## Idealized Optimum

次に、モデルが各文脈 $x$ に対して任意の分布 $p(\cdot\mid x)$ を独立に割り当てられる理想化された設定を考える。
このとき、各 $x$ についての最適化問題は

$$
\min_{p(\cdot\mid x)}
-\sum_y N(x,y)\log p(y\mid x)
\quad
\text{s.t.}\quad
\sum_y p(y\mid x)=1,\quad
p(y\mid x)\geq 0
$$

となる。

制約 $\sum_y p(y\mid x)=1$ をラグランジュ乗数 $\lambda$ を用いて組み込むと、ラグランジュ関数は

$$
\mathcal{J}_x
= -\sum_y N(x,y)\log p(y\mid x)
+\lambda\left(\sum_y p(y\mid x)-1\right)
$$

である。
正の確率が割り当てられる各 $p(y\mid x)$ について偏微分すると、

$$
\frac{\partial \mathcal{J}_x}{\partial p(y\mid x)}
= -\frac{N(x,y)}{p(y\mid x)}+\lambda
$$

であり、次のときに $0$ になる。

$$
p(y\mid x)=\frac{N(x,y)}{\lambda}\label{eq:lagrange}
$$

正規化条件を用いると、

$$
\sum_y p(y\mid x)
=\sum_y\frac{N(x,y)}{\lambda}
=\frac{1}{\lambda}\sum_y N(x,y)
=1
$$

より、

$$
\lambda=\sum_y N(x,y)
$$

である。
これを式$\ref{eq:lagrange}$ に代入すると、

$$
p^*(y\mid x)
=\frac{N(x,y)}{\sum_{y'}N(x,y')}
=\hat p(y\mid x)
$$

を得る。

つまり、学習データは各位置で one-hot target として与えられているが、同じ文脈 $x$ が複数回現れ、その後続トークンにばらつきがある場合、理想的な最適解は one-hot 分布ではなく、その文脈に対する経験的な次トークン分布になる。

## 注意点

上の議論は、各文脈 $x$ について独立に分布を選べる理想化された場合の話である。
実際のニューラル言語モデルではパラメータが文脈間で共有されているため、各 $x$ の分布を完全に独立に最適化できるわけではない。

また、softmax による有限の logit では $p(y\mid x)=0$ を厳密には実現できない。
そのため、$N(x,y)=0$ のトークンについては、確率を $0$ に近づける極限として理解するのが自然である。

さらに、有限のコーパスでは、長い文脈 $x$ がまったく同じ形で何度も現れるとは限らない。
上の導出は、同じ文脈ごとに経験分布を考えられる理想化された見方であり、実際の言語モデルではパラメータ共有によって多数の文脈から統計的な傾向を学習する。

## まとめ

- 各訓練例の target は one-hot である。
- しかし同じ文脈 $x$ ごとに集約すると、loss は経験分布 $\hat p(\cdot\mid x)$ とモデル分布 $p_\theta(\cdot\mid x)$ の cross entropy になる。
- 各文脈に対して自由に分布を割り当てられる理想化設定では、最適解は経験分布 $p^*(y\mid x)=\hat p(y\mid x)$ である。
- したがって、各サンプルの教師が one-hot であること自体は、次トークン分布を学習するという目的と矛盾しない。
