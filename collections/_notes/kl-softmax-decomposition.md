---
layout: single
classes: wide
toc: true
toc_sticky: true
breadcrumbs: false
title: "softmax(A+B+C) における A, B, C の寄与度の KL 分解"
use_math: true
header:
  teaser: /assets/img/kl-softmax-decomposition.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-07-10
excerpt: "softmax(A+B+C) の分布変化に対して、A, B, C がどれだけ寄与しているかを KL ベースで定量化するためのメモ。"
---
<!-- タイトル等は数式が使えないのでzではなくA, B, Cを用いる -->

$\bm{z}=\sum_k\bm{z}^{(k)}$ と分解したとき、

$$
\begin{equation}
D_{\mathrm{KL}}(\operatorname{softmax}(\bm{z})\|\operatorname{uniform})
% = \int_0^1 t\operatorname{Var}_{i\sim\bm{p}(t)}(z_i)\,dt
= \sum_k I_k \label{eq:kl-additive-decomposition-summary}
\end{equation}
$$

<!-- が成り立つ。
ただし、 -->

$$
\begin{align}
I_k
&= \int_0^1 t\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)\,dt\\
&= \bm{z}^{(k)\top} \left(\bm{p}(1) - \int_0^1 \bm{p}(t)\,dt\right)
\end{align}
$$

$$
\bm{p}(t) = \operatorname{softmax}( t\bm{z})
$$

またこの $I_k$ は、加法的に分解された logit 成分 $\bm{z}^{(k)}$に対する component-wise Integrated Gradients attribution、すなわち成分ごとの Integrated Gradients 寄与に対応する。


## 1. Bregman divergence

### 1.1. 定義

Bregman divergence は、任意の微分可能な凸関数 $F$ を用いて定義される、2つの点や確率分布の間の非対称な差異を測る尺度であり、以下のように定義される。

$$
\begin{equation}
D_F(\bm{p}\|\bm{q})
=F(\bm{p})-F(\bm{q})-\nabla_{\bm{p}} F(\bm{q})^\top(\bm{p}-\bm{q})
\label{eq:bregman-definition}
\end{equation}
$$

$F$ を点 $\bm{q}$ の周りで一次近似すると

$$
F(\bm{p}) \approx F(\bm{q}) + \nabla_{\bm{p}} F(\bm{q})^\top (\bm{p}-\bm{q})
$$

となる。したがって、Bregman divergence は、
> 点 $\bm{q}$ における $F$ の接平面による $F(\bm{p})$ の予測値と、実際の $F(\bm{p})$ の差

と解釈できる。

### 1.2. L2 距離

$$
F(\bm{p}) = \sum_i p_i^2
$$

とし、

$$
\begin{align}
\nabla_{\bm{p}} F(\bm{p}) = 2\bm{p}
\end{align}
$$

を利用すると

$$
\begin{align}
D_F(\bm{p}\|\bm{q})
&= F(\bm{p}) - F(\bm{q}) - \nabla_{\bm{p}} F(\bm{q})^\top (\bm{p}-\bm{q})
\quad(\because \text{式}\ref{eq:bregman-definition})\\
&= \sum_i p_i^2 - \sum_i q_i^2 - 2\bm{q}^\top (\bm{p}-\bm{q})\\
&= \sum_i p_i^2 - \sum_i q_i^2 - 2\sum_i q_i(p_i-q_i)\\
&= \sum_i p_i^2 - \sum_i q_i^2 - 2\sum_i p_iq_i + 2\sum_i q_i^2\\
&= \sum_i p_i^2 - 2\sum_i p_iq_i + \sum_i q_i^2\\
&= \sum_i (p_i-q_i)^2\\
&= \|\bm{p}-\bm{q}\|_2^2
\end{align}
$$

となり、Bregman divergence は L2 距離の二乗に対応する。



### 1.3. テイラーの定理の利用

$F: \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級の関数とする。
点 $\bm{a}\in\mathbb{R}^n$ の周りでの Taylor 展開は、積分剰余形を用いて以下のように表せる。

$$
\begin{align}
F(\bm{x}) = F(\bm{a}) + \nabla_{\bm{x}} F(\bm{a})^\top (\bm{x}-\bm{a}) + \int_0^1 (1-t)(\bm{x}-\bm{a})^\top \nabla^2_{\bm{x}} F(\bm{a}+t(\bm{x}-\bm{a}))(\bm{x}-\bm{a})\,dt
\end{align}
$$

したがって、

$$
\begin{align}
F(\bm{p}) = F(\bm{q}) + \nabla_{\bm{p}} F(\bm{q})^\top (\bm{p}-\bm{q}) + \int_0^1 (1-t)(\bm{p}-\bm{q})^\top \nabla^2_{\bm{p}} F(\bm{q}+t(\bm{p}-\bm{q}))(\bm{p}-\bm{q})\,dt
\end{align}
$$

より、

$$
\begin{align}
D_F(\bm{p}\|\bm{q})
&= F(\bm{p}) - F(\bm{q}) - \nabla_{\bm{p}} F(\bm{q})^\top (\bm{p}-\bm{q})
\quad(\because \text{式}\ref{eq:bregman-definition})\\
&= \int_0^1 (1-t)(\bm{p}-\bm{q})^\top \nabla^2_{\bm{p}} F(\bm{q}+t(\bm{p}-\bm{q}))(\bm{p}-\bm{q})\,dt
\label{eq:bregman-integral}
\end{align}
$$

## 2. 指数型分布族

### 2.1. 定義

ある確率分布が

$$
\begin{equation}
p_{\bm{\eta}}(x)
= h(x)\exp\left(\bm{\eta}^\top \bm{T}(x)-A(\bm{\eta})\right)
\label{eq:exponential-family}
\end{equation}
$$

と書けるとき、この分布族を指数型分布族という。

ここで各項は
- $\bm{\eta}$ : 自然パラメータ (natural parameter), $\bm{\eta}\in\mathbb{R}^k$
- $\bm{T}(x)$ : 十分統計量 (sufficient statistic), $\bm{T}:\mathcal{X}\to\mathbb{R}^k$
- $A(\bm{\eta})$ : 正規化項 (log-partition function), $A:\mathbb{R}^k\to\mathbb{R}$
- $h(x)$ : 基底測度 (base measure), $h:\mathcal{X}\to\mathbb{R}_{\ge 0}$

である。

### 2.2. 正規化項

$p_{\bm{\eta}}(x)$ が確率分布であるためには、以下が成り立つ必要がある。

$$
\begin{align}
\int_{\mathcal{X}} p_{\bm{\eta}}(x)\,dx
&=\int_{\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x)-A(\bm{\eta}))\,dx\\
&=\exp(-A(\bm{\eta}))\int_{\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))\,dx\\
&=1
\end{align}
$$

従って、

$$
\begin{align}
A(\bm{\eta}) = \log \int_{\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))\,dx
\end{align}
$$

離散分布の場合も同様に、

$$
\begin{align}
A(\bm{\eta}) = \log \sum_{x\in\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))
\end{align}
$$

### 2.3. 正規化項の微分

$A(\bm{\eta})$ の微分は、十分統計量の期待値に対応する。

$$
\begin{align}
\nabla_{\bm{\eta}} A(\bm{\eta})
&= \frac{1}{\int_{\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))\,dx} \int_{\mathcal{X}} h(x) \bm{T}(x)\exp(\bm{\eta}^\top \bm{T}(x))\,dx\\
&= \int_{\mathcal{X}} \bm{T}(x) \frac{h(x)\exp(\bm{\eta}^\top \bm{T}(x))}{\int_{\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))\,dx}\,dx\\
&= \int_{\mathcal{X}} \bm{T}(x) \frac{h(x)\exp(\bm{\eta}^\top \bm{T}(x))}{\exp(A(\bm{\eta}))}\,dx\\
&= \int_{\mathcal{X}} \bm{T}(x) h(x)\exp(\bm{\eta}^\top \bm{T}(x)-A(\bm{\eta}))\,dx\\
&= \int_{\mathcal{X}} \bm{T}(x) p_{\bm{\eta}}(x)\,dx\\
&= \mathbb{E}_{p_{\bm{\eta}}}[\bm{T}(X)]
\label{eq:log-partition-gradient}
\end{align}
$$

離散分布の場合も同様に、

$$
\begin{align}
\nabla_{\bm{\eta}} A(\bm{\eta})
&= \sum_{x\in\mathcal{X}} \bm{T}(x) p_{\bm{\eta}}(x)\\
&= \frac{\sum_{x\in\mathcal{X}} \bm{T}(x) h(x)\exp(\bm{\eta}^\top \bm{T}(x))}{\sum_{x\in\mathcal{X}} h(x)\exp(\bm{\eta}^\top \bm{T}(x))}\\
&= \frac{\sum_{x\in\mathcal{X}} \bm{T}(x) h(x)\exp(\bm{\eta}^\top \bm{T}(x))}{\exp(A(\bm{\eta}))}\\
&= \sum_{x\in\mathcal{X}} \bm{T}(x) h(x)\exp(\bm{\eta}^\top \bm{T}(x)-A(\bm{\eta}))\\
&= \sum_{x\in\mathcal{X}} \bm{T}(x) p_{\bm{\eta}}(x)\\
&= \mathbb{E}_{p_{\bm{\eta}}}[\bm{T}(X)]
\label{eq:log-partition-gradient-discrete}
\end{align}
$$


## 3. KL divergence

指数型分布族では、

$$
p_{\bm{\eta}}(x)
=h(x)\exp\left(\bm{\eta}^\top\bm{T}(x)-A(\bm{\eta})\right)
$$

であった。両辺の対数をとると、

$$
\begin{equation}
\log p_{\bm{\eta}}(x)
=\log h(x)+\bm{\eta}^\top\bm{T}(x)-A(\bm{\eta})
\label{eq:exponential-family-log-density}
\end{equation}
$$

ここで、同じ指数型分布族内の 2 つの分布
$p_{\bm{\eta}}(x)$ と $p_{\bm{\eta}'}(x)$ の KL divergence は、

$$
\begin{align}
D_{\mathrm{KL}}(p_{\bm{\eta}}\|p_{\bm{\eta}'})
&= \sum_x p_{\bm{\eta}}(x) \log \frac{p_{\bm{\eta}}(x)}{p_{\bm{\eta}'}(x)}\\
&= \sum_x p_{\bm{\eta}}(x)
\left(
\log h(x)+\bm{\eta}^\top\bm{T}(x)-A(\bm{\eta})
-\log h(x)-\bm{\eta}'^\top\bm{T}(x)+A(\bm{\eta}')
\right)
\quad(\because \text{式}\ref{eq:exponential-family-log-density})\\
&= \sum_x p_{\bm{\eta}}(x)
\left(
(\bm{\eta}-\bm{\eta}')^\top\bm{T}(x)
-A(\bm{\eta})+A(\bm{\eta}')
\right)\\
&= (\bm{\eta}-\bm{\eta}')^\top
\sum_x p_{\bm{\eta}}(x)\bm{T}(x)
-A(\bm{\eta})+A(\bm{\eta}')\\
&= (\bm{\eta}-\bm{\eta}')^\top
\mathbb{E}_{p_{\bm{\eta}}}[\bm{T}(X)]
-A(\bm{\eta})+A(\bm{\eta}')
\end{align}
$$

指数型分布族の特徴である式$\ref{eq:log-partition-gradient}$を用いると、

$$
\begin{align}
D_{\mathrm{KL}}(p_{\bm{\eta}}\|p_{\bm{\eta}'})
&= (\bm{\eta}-\bm{\eta}')^\top \nabla_{\bm{\eta}} A(\bm{\eta})
-A(\bm{\eta})+A(\bm{\eta}')
\quad(\because \text{式}\ref{eq:log-partition-gradient})\\
&= A(\bm{\eta}')-A(\bm{\eta})
-\nabla_{\bm{\eta}} A(\bm{\eta})^\top(\bm{\eta}'-\bm{\eta})\\
&= D_A(\bm{\eta}'\|\bm{\eta})
\label{eq:kl-bregman}
\end{align}
$$

となる。
したがって、指数型分布族における KL divergence は、正規化項 $A$ による Bregman divergence に対応する。


## 4. Softmax

### 4.1. 指数型分布族としての softmaxと、その KL divergence の経路積分表現

$\bm{p}=\operatorname{softmax}(\bm{z})$ は、指数型分布族の一種であり、
- 自然パラメータ $\bm{\eta}=\bm{z}$
- 十分統計量 $\bm{T}(i)=\bm{e}_i$（$\bm{e}_i$ は第 $i$ 標準基底ベクトル）
- 正規化項 $A(\bm{\eta})=\log \sum_j \exp(\eta_j)$
- 基底測度 $h(i)=1$

で表せる。
実際、

$$
\begin{align}
h(i)\exp\left(\bm{\eta}^\top \bm{T}(i)-A(\bm{\eta})\right)
&= 1 \cdot \exp\left(z_i - \log \sum_j \exp(z_j)\right)\\
&= \frac{\exp(z_i)}{\sum_j \exp(z_j)}\\
&= \operatorname{softmax}(\bm{z})_i
\end{align}
$$

となる。このとき、

$$
\begin{align}
\nabla_{\bm{z}} A(\bm{z}) 
&= \mathbb{E}_{\bm{z}}[\bm{T}(I)]\quad (\because \text{式}\ref{eq:log-partition-gradient-discrete})\\
&= \sum_i \operatorname{softmax}(\bm{z})_i \bm{e}_i\\
&= \operatorname{softmax}(\bm{z}) = \bm{p}
\label{eq:softmax-gradient}
\end{align}
$$

である。
$\sum_j \exp(z_j)=S$ とおくと、

$$
\begin{equation}
\frac{\partial S}{\partial z_i} = \exp(z_i) 
\end{equation}
$$

であり、

$$
\begin{align}
\nabla^2_{\bm{z}} A(\bm{z})
&= \frac{1}{S}\operatorname{diag}(\exp(\bm{z}))
-\frac{1}{S^2}\exp(\bm{z})\exp(\bm{z})^\top\\
&= \operatorname{diag}\left(\frac{\exp(\bm{z})}{S}\right)
-\frac{\exp(\bm{z})\exp(\bm{z})^\top}{S^2}\\
&= \operatorname{diag}(\bm{p}) - \bm{p}\bm{p}^\top
\label{eq:softmax-hessian}
\end{align}
$$


ここで、あるベースラインベクトル $\bm{\alpha}$ を用意し、

$$
\begin{equation}
\bm{p}(t) = \operatorname{softmax}(\bm{\alpha}+t\bm{z})
\label{eq:softmax-path}
\end{equation}
$$
とおく。
ここでは、$\bm{z}$ を加えた結果として得られる最終分布を基準に変化を評価するため、
$D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))$ を用いる。

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
&= D_A(\bm{\alpha}\|\bm{\alpha}+\bm{z})
\quad(\because \text{式}\ref{eq:kl-bregman})\\
&= \int_0^1 (1-t)(\bm{\alpha}-\bm{\alpha}-\bm{z})^\top \nabla^2 A(\bm{\alpha}+\bm{z}+t(\bm{\alpha}-\bm{\alpha}-\bm{z}))(\bm{\alpha}-\bm{\alpha}-\bm{z})\,dt\\
&= \int_0^1 (1-t)(-\bm{z})^\top \nabla^2 A(\bm{\alpha}+(1-t)\bm{z})(-\bm{z})\,dt\\
&= \int_0^1 (1-t)\bm{z}^\top \nabla^2 A(\bm{\alpha}+(1-t)\bm{z})\bm{z}\,dt\\
&= \int_0^1 t\bm{z}^\top \nabla^2 A(\bm{\alpha}+t\bm{z})\bm{z}\,dt
\label{eq:kl-path-hessian}
\end{align}
$$

ここで、

$$
\begin{align}
\bm{z}^\top \nabla^2 A(\bm{\alpha}+t\bm{z})\bm{z}
&= \bm{z}^\top
\left(\operatorname{diag}(\bm{p}(t))-\bm{p}(t)\bm{p}(t)^\top\right)\bm{z}
\quad(\because \text{式}\ref{eq:softmax-hessian})\\
&= \sum_i z_i^2p(t)_i-\left(\sum_i z_ip(t)_i\right)^2\\
&= \mathbb{E}_{i\sim\bm{p}(t)}[z_i^2]
-\left(\mathbb{E}_{i\sim\bm{p}(t)}[z_i]\right)^2\\
&= \operatorname{Var}_{i\sim\bm{p}(t)}(z_i)
\label{eq:softmax-path-variance}
\end{align}
$$

より、

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
&= \int_0^1 t\operatorname{Var}_{i\sim\bm{p}(t)}(z_i)\,dt
\quad(\because \text{式}\ref{eq:kl-path-hessian},\ \text{式}\ref{eq:softmax-path-variance})
\label{eq:kl-path-variance}
\end{align}
$$

なお、

$$
\begin{align}
\frac{d}{dt}p(t)_i
&= \frac{d}{dt}\operatorname{softmax}(\alpha_i+t z_i)\\
&= \frac{d}{dt}\frac{\exp(\alpha_i+t z_i)}{\sum_j \exp(\alpha_j+t z_j)}\\
&= \frac{z_i\exp(\alpha_i+t z_i)\left(\sum_j \exp(\alpha_j+t z_j)\right) - \exp(\alpha_i+t z_i)\left(\sum_j z_j\exp(\alpha_j+t z_j)\right)}{\left(\sum_j \exp(\alpha_j+t z_j)\right)^2}\\
&= z_i p(t)_i - p(t)_i \frac{\sum_j z_j\exp(\alpha_j+t z_j)}{\sum_j \exp(\alpha_j+t z_j)}\\
&= z_i p(t)_i - p(t)_i \sum_j z_j p(t)_j\\
&= p(t)_i\left(z_i - \sum_j z_j p(t)_j\right)\\
\end{align}
$$

より、

$$
\begin{align}
\frac{d}{dt}\mathbb{E}_{i\sim\bm{p}(t)}[z_i]
&= \frac{d}{dt}\sum_i z_i p(t)_i\\
&= \sum_i z_i \frac{d}{dt}p(t)_i\\
&= \sum_i z_i p(t)_i\left(z_i - \sum_j z_j p(t)_j\right)\\
&= \sum_i z_i^2 p(t)_i - \sum_i z_i p(t)_i \sum_j z_j p(t)_j\\
&= \mathbb{E}_{i\sim\bm{p(t)}}[z_i^2] - \left(\mathbb{E}_{i\sim\bm{p(t)}}[z_i]\right)^2\\
&= \operatorname{Var}_{i\sim\bm{p(t)}}(z_i)
\label{eq:kl-path-variance-derivative}
\end{align}
$$

であるから、式 \ref{eq:kl-path-variance} は

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
&= \int_0^1 t\left(\frac{d}{dt}\mathbb{E}_{i\sim\bm{p}(t)}[z_i]\right)\,dt\\
&= \mathbb{E}_{i\sim\bm{p}(1)}[z_i] - \int_0^1 \mathbb{E}_{i\sim\bm{p}(t)}[z_i]\,dt\\
&= \bm{z}^\top \bm{p}(1) - \int_0^1 \bm{z}^\top \bm{p}(t)\,dt\\
&= \bm{z}^\top \left(\bm{p}(1) - \int_0^1 \bm{p}(t)\,dt\right)
\end{align}
$$

とも書ける。

### 4.2. $\bm{z}$ の分解

次に、$\bm{z}=\sum_k\bm{z}^{(k)}$ と分解する ($\bm{z}^{(k)}\in \mathbb{R}^n$)。
$z_i=\sum_k z_i^{(k)}$ なので、

$$
\begin{align}
\operatorname{Var}_{i\sim\bm{p}(t)}(z_i)
&=\operatorname{Cov}_{i\sim\bm{p}(t)}\left(\sum_k z_i^{(k)},z_i\right)\\
&=\sum_k\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)
\label{eq:variance-component-decomposition}
\end{align}
$$

である。したがって、

$$
\begin{equation}
I_k
=
\int_0^1
t
\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)
\,dt
\label{eq:component-contribution}
\end{equation}
$$

と定義すれば、

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
&= \int_0^1 t\operatorname{Var}_{i\sim\bm{p}(t)}(z_i)\,dt
\quad(\because \text{式}\ref{eq:kl-path-variance})\\
&= \int_0^1 t\sum_k \operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)\,dt
\quad(\because \text{式}\ref{eq:variance-component-decomposition})\\
&= \sum_k \int_0^1 t\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)\,dt\\
&= \sum_k I_k
\quad(\because \text{式}\ref{eq:component-contribution})
\label{eq:kl-additive-decomposition}
\end{align}
$$

が厳密に成り立つ。

特に$\bm{\alpha}=\bm{0}$ とすると、冒頭の式 \ref{eq:kl-additive-decomposition-summary} が得られる。

また、式\ref{eq:kl-path-variance-derivative} と同様に考えると、

$$
\begin{align}
\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)
&= \frac{d}{dt}\mathbb{E}_{i\sim\bm{p}(t)}[z_i^{(k)}]
\end{align}
$$

となり、

$$
\begin{align}
I_k 
&= \int_0^1 t\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)\,dt\\
&= \int_0^1 t\frac{d}{dt}\mathbb{E}_{i\sim\bm{p}(t)}[z_i^{(k)}]\,dt\\
&= \mathbb{E}_{i\sim\bm{p}(1)}[z_i^{(k)}] - \int_0^1 \mathbb{E}_{i\sim\bm{p}(t)}[z_i^{(k)}]\,dt\\
&= \bm{z}^{(k)\top} \left(\bm{p}(1) - \int_0^1 \bm{p}(t)\,dt\right)
\end{align}
$$

とも書ける。



#### 4.2.1. 解釈

この定義では、各 $I_k$ は「その成分が、全体 $\bm{z}$ と同じ方向にどれだけ softmax 分布を動かしているか」を表す符号付き寄与と解釈できる。経路上の各点で $\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)$ が正であれば、その成分は全体の分布変化を強める方向に働き、負であれば打ち消す方向に働く。

ここで、$\bm{p}:=\bm{p}(1)$、$\bm{q}:=\bm{p}(0)$ とおくと、

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}\|\bm{q})
&= \sum_i p_i\log\frac{p_i}{q_i}\\
&= \sum_i p_i\log p_i-\sum_i p_i\log q_i\\
&= \sum_k I_k
\quad(\because \text{式}\ref{eq:kl-additive-decomposition})
\end{align}
$$

である。したがって、

$$
\begin{align}
\sum_i p_i\log p_i
&=
\sum_i p_i\log q_i
+
\sum_k I_k
\end{align}
$$

と書ける。すなわち、各 $I_k$ は、ベースライン $\bm{q}$ の対数確率を
最終分布 $\bm{p}$ のもとで評価した値から、最終分布自身の対数確率を
同じく $\bm{p}$ のもとで評価した値へ至る差を加法分解している。

特に、$\bm{\alpha}=c\bm{1}$（例えば $\bm{\alpha}=\bm{0}$）なら、
ベースライン分布は $n$ 個のカテゴリ上の一様分布
$q_i=1/n$ となる。このとき、

$$
\begin{align}
\sum_i p_i\log p_i
&=\sum_i p_i\log (1/n)
+
\sum_k I_k\\
&=-\log n + \sum_k I_k
\end{align}
$$

左辺は $\bm{p}$ の負のエントロピー $-H(\bm{p})$ なので、

$$
\begin{equation}
H(\bm{p})
=
\log n-\sum_k I_k
\label{eq:entropy-contribution}
\end{equation}
$$

となる。したがって、$\sum_k I_k$ は、一様分布の最大エントロピー
$\log n$ から、$\bm{z}$ を加えることでどれだけエントロピーが減少し、
分布が集中したかを表す。
各 $I_k$ はこのエントロピー減少への符号付き寄与と解釈でき、
$I_k>0$ なら分布を集中させる側、$I_k<0$ なら集中を打ち消して
一様分布に近づける側に寄与している。

#### 4.2.2. Integrated Gradients との接続
ベクトル $\bm{x}\in\mathbb{R}^n$ を関数$F$ に入力したときの、$i$番目の成分の寄与度を計算する Integrated Gradients は、ベースラインベクトル $\bm{x}'$ を用いて

$$
\begin{equation}
\operatorname{IG}_i(\bm{x}) = (x_i-x_i')\int_0^1 \left.\frac{\partial F(\bm{u})}{\partial u_i}\right|_{\bm{u}=\bm{x}' + t(\bm{x}-\bm{x}')}\,dt
\label{eq:integrated-gradients}
\end{equation}
$$

で表される。

ここでは、

$$
\begin{align}
\bm{p}(\bm{s}) = \operatorname{softmax}\left(\bm{\alpha}+\sum_j s_j\bm{z}^{(j)}\right)
\label{eq:gated-softmax}
\end{align}
$$
とおき、

- $\bm{x}$ を $\bm{s}$
- $\bm{x}'$ を $\bm{0}$
- $F(\bm{u})$ を $D_{\mathrm{KL}}(\bm{p}(\bm{u})\|\bm{p }(\bm{0}))$

とした以下を考える

$$
\begin{align}
\operatorname{IG}_k(\bm{s})
&= (s_k-0)\int_0^1 \frac{\partial D_{\mathrm{KL}}(\bm{p}(\bm{u})\|\bm{p}(\bm{0}))}{\partial u_k}\Big|_{\bm{u}=\bm{0} + t(\bm{s}-\bm{0})}\,dt
\quad(\because \text{式}\ref{eq:integrated-gradients})\\
&= s_k\int_0^1 \frac{\partial D_{\mathrm{KL}}(\bm{p}(\bm{u})\|\bm{p}(\bm{0}))}{\partial u_k}\Big|_{\bm{u}=t\bm{s}}\,dt\\
&= s_k\int_0^1 \frac{\partial D_A(\bm{\alpha}\|\bm{\alpha}+\sum_j u_j\bm{z}^{(j)})}{\partial u_k}\Big|_{\bm{u}=t\bm{s}}\,dt
\quad(\because \text{式}\ref{eq:kl-bregman})
\end{align}
$$




まず、$D_A(\bm{\alpha}\|\bm{\eta})$ について、$\bm{\eta}$ で微分すると、

$$
\begin{align}
\nabla_{\bm{\eta}} D_A(\bm{\alpha}\|\bm{\eta})
&= \nabla_{\bm{\eta}} \left(A(\bm{\alpha})-A(\bm{\eta})-\nabla A(\bm{\eta})^\top(\bm{\alpha}-\bm{\eta})\right)\\
&= 0 - \nabla A(\bm{\eta}) - \nabla^2 A(\bm{\eta})(\bm{\alpha}-\bm{\eta}) + \nabla A(\bm{\eta})\\
&= \nabla^2 A(\bm{\eta})(\bm{\eta}-\bm{\alpha})
\label{eq:bregman-gradient-second-argument}
\end{align}
$$

となる。
次に、$\bm{\eta}$ が $\bm{u} \in \mathbb{R}^n$ に依存する場合を考え、$u_k$ ($\bm{u}$の$k$番目の成分)で微分すると、

$$
\begin{align}
\frac{\partial}{\partial u_k} D_A(\bm{\alpha}\|\bm{\eta}(\bm{u}))
&= \left(\frac{\partial \bm{\eta}(\bm{u})}{\partial u_k}\right)^\top \nabla_{\bm{\eta}} D_A(\bm{\alpha}\|\bm{\eta}(\bm{u}))\\
&= \left(\frac{\partial \bm{\eta}(\bm{u})}{\partial u_k}\right)^\top \nabla^2 A(\bm{\eta}(\bm{u}))(\bm{\eta}(\bm{u})-\bm{\alpha})
\quad(\because \text{式}\ref{eq:bregman-gradient-second-argument})
\end{align}
$$

である。ここで、$\bm{\eta}(\bm{u})=\bm{\alpha}+\sum_j u_j\bm{z}^{(j)}$ とおくと、

$$
\begin{align}
\frac{\partial \bm{\eta}(\bm{u})}{\partial u_k} = \bm{z}^{(k)}
\end{align}
$$

より、

$$
\begin{align}
\frac{\partial}{\partial u_k} D_A(\bm{\alpha}\|\bm{\eta}(\bm{u}))
&= \left(\bm{z}^{(k)}\right)^\top \nabla^2 A(\bm{\eta}(\bm{u}))(\bm{\eta}(\bm{u})-\bm{\alpha})\\
\end{align}
$$

したがって、

$$
\begin{align}
\operatorname{IG}_k(\bm{s})
&= s_k\int_0^1 \left(\bm{z}^{(k)}\right)^\top \nabla^2 A(\bm{\eta}(\bm{u}))(\bm{\eta}(\bm{u})-\bm{\alpha})\Big|_{\bm{u}=t\bm{s}}\,dt
\quad(\because \text{式}\ref{eq:bregman-gradient-second-argument})\\
&= s_k\int_0^1 \left(\bm{z}^{(k)}\right)^\top \nabla^2 A(\bm{\eta}(t\bm{s}))(\bm{\eta}(t\bm{s})-\bm{\alpha})\,dt\\
\end{align}
$$

$\bm{s}=\bm{1}$を代入すると、

$$
\begin{align}
\operatorname{IG}_k(\bm{1})
&= \int_0^1 \left(\bm{z}^{(k)}\right)^\top \nabla^2 A(\bm{\eta}(t\bm{1}))(\bm{\eta}(t\bm{1})-\bm{\alpha})\,dt\\
&= \int_0^1 \left(\bm{z}^{(k)}\right)^\top \nabla^2 A\left(\bm{\alpha}+t\sum_j \bm{z}^{(j)}\right)\left(\bm{\alpha}+t\sum_j \bm{z}^{(j)}-\bm{\alpha}\right)\,dt\\
&= \int_0^1 t\left(\bm{z}^{(k)}\right)^\top \nabla^2 A\left(\bm{\alpha}+t\sum_j \bm{z}^{(j)}\right)\left(\sum_j \bm{z}^{(j)}\right)\,dt\\
&= \int_0^1 t\operatorname{Cov}_{i\sim\bm{p}(t)}(z_i^{(k)},z_i)\,dt\\
&= I_k
\quad(\because \text{式}\ref{eq:component-contribution})
\label{eq:ig-equals-component-contribution}
\end{align}
$$

このことから、まとめると、以下が成り立つ

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
&= \sum_k I_k
\quad(\because \text{式}\ref{eq:kl-additive-decomposition})\\
&= \sum_k \operatorname{IG}_k(\bm{1})
\quad(\because \text{式}\ref{eq:ig-equals-component-contribution})
\label{eq:ig-completeness}
\end{align}
$$


### 4.3. Leave-one-out KL

成分 $\bm{z}^{(k)}$ を除いたときの分布を

$$
\begin{equation}
\bm{p}_{-k}
=
\operatorname{softmax}\left(
\bm{z}-\bm{z}^{(k)}
\right)
\label{eq:loo-distribution}
\end{equation}
$$

とおき、元の分布
$\bm{p}=\operatorname{softmax}(\bm{z})$ との差

$$
\begin{equation}
L_k
:=
D_{\mathrm{KL}}(\bm{p}\|\bm{p}_{-k})
\label{eq:loo-contribution}
\end{equation}
$$

を成分 $\bm{z}^{(k)}$ の leave-one-out 寄与と定義する。
これは上記の議論で、ベースラインを
$\bm{z}-\bm{z}^{(k)}$、そこに加えるベクトルを
$\bm{z}^{(k)}$ とした場合に対応する。

このときの経路を

$$
\begin{equation}
\bm{p}_k(t)
=
\operatorname{softmax}\left(
\bm{z}-\bm{z}^{(k)}+t\bm{z}^{(k)}
\right)
\label{eq:loo-path}
\end{equation}
$$

とおけば、$$\bm{p}_k (0)=\bm{p}_{-k}, \bm{p}_k (1)=\bm{p}$$ である。したがって、

$$
\begin{align}
L_k
&=
\int_0^1
t\operatorname{Var}_{i\sim\bm{p}_k(t)}
\left(z_i^{(k)}\right)
\,dt
\quad(\because \text{式}\ref{eq:kl-path-variance})
\label{eq:loo-path-variance}
\end{align}
$$

が成り立つ。$L_k\ge 0$ であり、成分 $\bm{z}^{(k)}$ を除くことで
softmax 分布がどれだけ変化するかを表す。

ただし、$L_k$ は各成分を異なるベースライン
$\bm{z}-\bm{z}^{(k)}$ から評価した量である。そのため、一般には

$$
D_{\mathrm{KL}}(\bm{p}(1)\|\bm{p}(0))
\neq
\sum_k L_k
$$

であり、前節の $I_k$ のような KL divergence の加法分解にはならない。



#### 4.3.1. Integrated Gradients との接続

他の成分を固定し、$\bm{z}^{(k)}$ だけを制御するスカラーのゲート $s$ を導入する。
ここでは、

$$
\begin{align}
\bm{p}_k(s)
=
\operatorname{softmax}\left(
\bm{z}-\bm{z}^{(k)}+s\bm{z}^{(k)}
\right)
\label{eq:loo-gated-softmax}
\end{align}
$$

とおき、

- $x$ を $s$
- $x'$ を $0$
- $F(u)$ を $D_{\mathrm{KL}}(\bm{p}_k(u)\|\bm{p}_k(0))$

とした 1 次元の Integrated Gradients を考える。このとき、

$$
\begin{align}
\operatorname{IG}^{\mathrm{LOO}}_k(s)
&=
(s-0)\int_0^1
\left.
\frac{\partial
D_{\mathrm{KL}}(\bm{p}_k(u)\|\bm{p}_k(0))}
{\partial u}
\right|_{u=ts}
\,dt
\quad(\because \text{式}\ref{eq:integrated-gradients})
\end{align}
$$

である。

ここで、

$$
\begin{equation}
\bm{\beta}_k
:=
\bm{z}-\bm{z}^{(k)}
\label{eq:loo-baseline-logit}
\end{equation}
$$

とおくと、

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}_k(u)\|\bm{p}_k(0))
&=
D_A\left(
\bm{\beta}_k\middle\|
\bm{\beta}_k+u\bm{z}^{(k)}
\right)
\quad(\because \text{式}\ref{eq:kl-bregman})
\end{align}
$$

である。前節と同様に、Bregman divergence を $u$ で微分すると、

$$
\begin{align}
\frac{\partial}{\partial u}
D_A\left(
\bm{\beta}_k\middle\|
\bm{\beta}_k+u\bm{z}^{(k)}
\right)
&=
\left(\bm{z}^{(k)}\right)^\top
\nabla^2 A\left(
\bm{\beta}_k+u\bm{z}^{(k)}
\right)
\left(u\bm{z}^{(k)}\right)
\quad(\because \text{式}\ref{eq:bregman-gradient-second-argument})\\
&=
u\operatorname{Var}_{i\sim\bm{p}_k(u)}
\left(z_i^{(k)}\right)
\label{eq:loo-kl-gradient}
\end{align}
$$

となる。したがって、$s=1$ を代入すると、

$$
\begin{align}
\operatorname{IG}^{\mathrm{LOO}}_k(1)
&=
\int_0^1
t\operatorname{Var}_{i\sim\bm{p}_k(t)}
\left(z_i^{(k)}\right)
\,dt
\quad(\because \text{式}\ref{eq:loo-kl-gradient})\\
&=L_k
\quad(\because \text{式}\ref{eq:loo-path-variance})
\label{eq:loo-ig-equals-contribution}
\end{align}
$$

となる。まとめると、

$$
\begin{align}
D_{\mathrm{KL}}(\bm{p}\|\bm{p}_{-k})
&=L_k
\quad(\because \text{式}\ref{eq:loo-contribution})\\
&=\operatorname{IG}^{\mathrm{LOO}}_k(1)
\quad(\because \text{式}\ref{eq:loo-ig-equals-contribution})
\end{align}
$$

が成り立つ。つまり、$L_k$ は、他の成分をすべて固定し、
$\bm{z}^{(k)}$ だけを除去状態から元の状態まで戻す 1 次元の
Integrated Gradients と解釈できる。
