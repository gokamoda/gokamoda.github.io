---
layout: splash
title: "RMSNorm and LayerNorm"
use_math: true
header:
  teaser: /assets/img/centering_matrix.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2024-07-12 # moved from DissectingGPT
excerpt: "Relationship between RMSNorm and LayerNorm and rewritten LayerNorm using centering matrix."
---
## Preparation

Let $\mu(\bm{x}): \mathbb{R}^{d}\rightarrow \mathbb{R}$ be a function that returns element-wise mean of a row-vector $\bm{x} \in \mathbb{R}^{d}$:

$$
\begin{align}
\mu(\bm{x}) 
&= \frac{1}{d}\sum_{i=1}^d \bm{x}_i\\
&=\frac{1}{d}\bm{x}\cdot \bm{1}\\
&=\frac{1}{d}\bm{x}\bm{1}^\top
\end{align}
$$


Let $c(\bm{x}): \mathbb{R}^{d}\rightarrow \mathbb{R}^{d}$ be centering function, that subtracts the element-wise mean from each element of $\bm{x}$:

$$
\begin{aligned}
	c(\bm{x})&=\bm{x} - \mu(\bm{x})\bm{1}\\
	&= \bm{x} - \frac{1}{d}\bm{x}\bm{1}^\top\bm{1}\\
	&= \bm{x} \left(1 - \frac{1}{d}\bm{1}^\top\bm{1}\right)
\end{aligned}
$$

By the way, \(I - \frac{1}{d}\bm{1}^\top\bm{1}\) is called the 
<a href="https://en.wikipedia.org/wiki/Centering_matrix">centering matrix</a>.




Let $\text{RMS}(\bm{x}): \mathbb{R}^{d}\rightarrow \mathbb{R}$ be a function that returns the element-wise RMS (root mean square):

$$
\begin{align}
	\text{RMS}(\bm{x})&=\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}\\
	&=\frac{\sqrt{\sum_{i=1}^d x_i^2}}{\sqrt{d}}\\
	&=\frac{||\bm{x}||_2}{\sqrt{d}}
\end{align}
$$

Let $\text{MS}(\bm{x}): \mathbb{R}^{d}\rightarrow \mathbb{R}$ be a function that returns the squared RMS (root mean square):

$$
\begin{align}
	\text{MS}(\bm{x})&=\text{RMS}(\bm{x})^2\\
	&=\frac{||\bm{x}||_2^2}{d}\\
	&=\frac{1}{d}\sum_{i=1}^d x_i^2\\
\end{align}
$$


Let $\text{Var}(\bm{x}): \mathbb{R}^{d}\rightarrow \mathbb{R}$ be a function that returns element-wise variance:

$$
\begin{align}
\text{Var}(\bm{x}) &= \frac{1}{d}\sum_{i=1}^d (x_i - \mu(\bm{x}))^2\\
&=\frac{1}{d}\sum_{i=1}^d (\bm{x} - \mu(\bm{x})\bm{1})^2_i\\
&=\frac{1}{d}\sum_{i=1}^d c(\bm{x})_i^2\\
&=\text{MS}(c(\bm{x}))
\end{align}
$$


## RMSNorm
[PyTorch: RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)

$$
\text{RMSNorm}(\bm{x}) = \frac{\bm{x}}{\sqrt{\text{MS}(\bm{x})+\varepsilon}}\odot \bm{\gamma}
$$

Here, $\odot$ is element-wise multiplication, $\bm{\gamma}\in \mathbb{R}^d$ is a learnable weight vector, and $\varepsilon$ is a small constant for numerical stability.


## LayerNorm
[PyTorch: LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

In the original form:

$$
\text{LayerNorm}(\bm{x}) = \frac{\bm{x} - \mu(\bm{x})\bm{1}}{\sqrt{\text{Var}(\bm{x})+\varepsilon}}\odot \bm{\gamma} +\bm{\beta}
$$

This can be rewritten using the centering function $c(\bm{x})$ and the MS function $\text{MS}(\bm{x})$ as follows:

$$
\begin{aligned}
\text{LayerNorm}(\bm{x}) &= \frac{c(\bm{x})}{\sqrt{\text{MS}(c(\bm{x}))+\varepsilon}}\odot \bm{\gamma} +
\bm{\beta}\\
\end{aligned}
$$

Thus, the following holds: `LayerNorm is equal to "centering" → RMSNorm → "add bias"`

$$
\text{LayerNorm}(\bm{x}) = \text{RMSNorm}(c(\bm{x})) + \bm{\beta}
$$

Also, element-wise multiplication of $\bm{\gamma}$ can be expressed as matrix multiplication of $\text{diag}(\bm{\gamma})$.
Therefore, LayerNorm can be rewritten as:

$$
\begin{align}
\text{LayerNorm}(\bm{x}) &= \frac{1}{\sqrt{\text{Var}(\bm{x})+\varepsilon}}\left(\bm{x} - \mu(\bm{x})\bm{1}\right)\odot \bm{\gamma} + \bm{\beta}\\
&= \frac{\bm{x}}{\sqrt{\text{Var}(\bm{x})+\varepsilon}}\left(I - \frac{1}{d}\bm{1}^\top\bm{1}\right)\text{diag}(\bm{\gamma}) + \bm{\beta}\\
\end{align}
$$

Thus only non-linear operation in LayerNorm is the division by $\sqrt{\text{Var}(\bm{x})+\varepsilon}$.




