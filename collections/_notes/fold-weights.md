---
layout: splash
title: "Folding weights in transformers"
use_math: true
header:
  teaser: 
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2025-05-12 # moved from DissectingGPT
excerpt: ""
---

This reformulation of LayerNorm and Self-Attention is used in our paper:
<div>
  <ul>
    <li>
      {% assign publication = site.pubInternationalConferences | where: "title", "Weight-based Analysis of Detokenization in Language Models: Understanding the First Stage of Inference Without Inference" | first %}
      {% include pub-apa-international-conf.html  publication=publication %}
    </li>
  </ul>
</div>


## Notation

$$
\begin{alignat}{4}
    &\bm{X} &:= &
    \begin{bmatrix}
        \bm{x}_1\\
        \vdots\\
        \bm{x}_n
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{n \times d}\\
    &\bm{W}^O &:= &
    \begin{bmatrix}
        \bm{W}^O_1\\
        \vdots\\
        \bm{W}^O_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d \times d}\\
    &\bm{W}^Q &:= &
    \begin{bmatrix}
        \bm{W}^Q_1 & \cdots & \bm{W}^Q_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d \times d}& \label{eq:wq_split}\\
    &\bm{W}^K &:= &
    \begin{bmatrix}
        \bm{W}^K_1 & \cdots & \bm{W}^K_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d \times d}& \label{eq:wk_split}\\
    &\bm{W}^V &:= &
    \begin{bmatrix}
        \bm{W}^V_1 & \cdots & \bm{W}^V_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d \times d}&\label{eq:wv_split}\\
    &\bm{b}^Q &:= &
    \begin{bmatrix}
        \bm{b}^Q_1 & \cdots & \bm{b}^Q_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d}& \label{eq:bq_split}\\
    &\bm{b}^K &:= &
    \begin{bmatrix}
        \bm{b}^K_1 & \cdots & \bm{b}^K_H
    \end{bmatrix}
    &\hspace{1em}\in& \mathbb{R}^{d}& \label{eq:bk_split}\\
    &\bm{b}^V &:= &
    \begin{bmatrix}
        \bm{b}^V_1 & \cdots & \bm{b}^V_H
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d}& \\
    &\bm{I} &:= &
    \begin{bmatrix}
        1 & 0 & \cdots & 0 \\
        0 & 1 & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & 1 \\
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d\times d}& \\
    &\bm{1} &:= &
    \begin{bmatrix}
        1 & \cdots & 1
    \end{bmatrix}
    &\hspace{1em}\in &\mathbb{R}^{d}
\end{alignat}
$$

## Original LayerNorm

Layer Normalization can be expressed as follows (org stands for original):

$$
\begin{alignat}{3}
    &\text{LN}(\bm{x}) &:=&\ \frac{\bm{x}-\mu(x)\bm{1}}{\sigma(\bm{x})}\odot\bm{\gamma} + \bm{\beta}&\hspace{1em}\in&\mathbb{R}^d\\
    &\bm{x} &:=&\ 
    \begin{bmatrix}
        x^{(1)} & \cdots & x^{(d)}
    \end{bmatrix}
    &\hspace{1em}\in&\mathbb{R}^d\\
    &\mu(\bm{x}) &:=&\ \frac{1}{d}\sum_kx^{(k)}&\hspace{1em}\in&\mathbb{R}\\
    &\sigma(\bm{x}) &:=&\ \sqrt{\frac{1}{d}\sum_k^d\left(x^{(k)}-\mu(\bm{x})\right)^2+\epsilon}&\hspace{1em}\in&\mathbb{R}
\end{alignat}
$$


Now, $\mu(\bm{x})$ can be reformulated as follows:

$$
\begin{align}
    \mu(\bm{x})\bm{1}
    &=\frac{1}{d}\left(\sum_kx^{(k)}\right)\bm{1}\\
    &=\frac{1}{d}\left(\bm{x}\bm{1}^\top\right)\bm{1}\\
    &=\bm{x}\left(\frac{1}{d}\bm{1}^\top\bm{1}\right)
\end{align}
$$

Thus $\text{LN}_{\text{org}}$ can be reformulated as follows.

$$
\begin{align}
    \text{LN}(\bm{x}) 
    &= \frac{\bm{x}-\mu(\bm{x})\bm{1}}{\sigma(\bm{x})}\odot\bm{\gamma} + \bm{\beta}\\
    &= \frac{1}{\sigma(\bm{x})} \left(\bm{x}-\bm{x}\left(\frac{1}{d}\bm{1}^\top\bm{1}\right)\right) \diag{\bm\gamma}+ \bm{\beta}\\
    &= \frac{\bm{x}}{\sigma(\bm{x})}\left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm\gamma}+ \bm{\beta}
\end{align}
$$

## Original Self-Attention

Let query, key, value transformations of each head $h$ be expressed as follows:

$$
\begin{align}
    \bm{q}_h(\bm{x}) &:= \bm{x}\bm{W}_h^Q + \bm{b}_h^Q\\
    \bm{k}_h(\bm{x}) &:= \bm{x}\bm{W}_h^K + \bm{b}_h^K\\
    \bm{v}_h(\bm{x}) &:= \bm{x}\bm{W}_h^V + \bm{b}_h^V\\
\end{align}
$$

Let attention weight from token position $i$ to $j$  ($i \ge j$) in head $h$ be expressed as follows:

$$
\alpha_{i, j, h} = \underset{\bm{x}_j,\bm{x}_j \in \bm{X}, j \leq i}{\text{softmax}}\frac{\bm{q}_h(\bm{x}_i)\bm{k}_h(\bm{x}_j)^\top}{\sqrt{d'}}
$$

where $d' = d/H$ is the dimension of each head.


The output of Attention layer of an causal model at position $i$ can be expressed as follows:

$$
\begin{align}
    \text{ATTN}(i, \bm{X})
        &:=\left[\text{head}_1(i, \bm{X})\hspace{0.5em}\cdots\hspace{0.5em}\text{head}_H(i, \bm{X})\right]
            \bm{W}^O + \bm{b}^O\\
        &=\sum_{h=1}^H \text{head}_h(i, \bm{X})\bm{W}^O_h + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \bm{v}_h(\bm{x}_j)\right)\bm{W}^O_h + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \left(\bm{x}_j\bm{W}^V_h + \bm{b}^V_h\right)\right)\bm{W}^O_h + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h}\bm{b}^V_h\right)\bm{W}^O_h + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H\bm{b}^V_h\bm{W}^O_h + \bm{b}^O\hspace{0.5em} \left(\because \sum_j \alpha_{i, j, h} = 1\right)\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \bm{b}^V\bm{W}^O + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h} \bm{x}_j\bm{W}^{VO}_h + \bm{b}^{VO}
\end{align}
$$

where   

$$
\begin{align}
    \bm{W}^{VO}_h &:= \bm{W}^V_h\bm{W}^O_h &\hspace{1em}\in&\mathbb{R}^{d \times d}\\
    \bm{b}^{VO} &:= \bm{b}^V\bm{W}^O + \bm{b}^O &\hspace{1em}\in&\mathbb{R}^{d}\\
\end{align}
$$

## Reformulating LayerNorm and Self-Attention

LayerNorm is always followed by a linear transformation in transformers.
Thus, we can fold the weights of LayerNorm into the weights of the following linear transformation.

For example, in the case of LayerNorm followed by query transformation, we can fold the weights as follows:

$$
\begin{align}
    \bm{q}_h(\text{LN}(\bm{x}))
        &= \text{LN}(\bm{x})\bm{W}^Q_h + \bm{b}^Q_h\\
        &= \left(\frac{\bm{x}}{\sigma(\bm{x})}\left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}+ \bm{\beta}\right)\bm{W}^Q_h + \bm{b}^Q_h\\
        &= \frac{\bm{x}}{\sigma(\bm{x})}\left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}\bm{W}^Q_h + \bm{\beta}\bm{W}^Q_h + \bm{b}^Q_h\\
        &= \overset{\text{new}}{\text{LN}}(\bm{x})\overset{\text{new}}{\bm{W}^Q_h} + \overset{\text{new}}{\bm{b}^Q_h}
\end{align}
$$

where

$$
\begin{align}
    \overset{\text{new}}{\text{LN}}(\bm{x}) &:= \frac{\bm{x}}{\sigma(\bm{x})} &\hspace{1em}\in&\mathbb{R}^d\\
    \overset{\text{new}}{\bm{W}^Q_h} &:= \left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}\bm{W}^Q_h &\hspace{1em}\in&\mathbb{R}^{d \times d}\\
    \overset{\text{new}}{\bm{b}^Q_h} &:= \bm{\beta}\bm{W}^Q_h + \bm{b}^{\{Q, K\}}_h &\hspace{1em}\in&\mathbb{R}^{d}\\
\end{align}
$$

The same can be done for key and value transformations, and thus LayerNorm followed by self-attention can be reformulated as follows:

$$
\begin{align}
    \overset{\text{new}}{\text{LN}}(\bm{x}) &:= \frac{\bm{x}}{\sigma(\bm{x})} &\hspace{1em}\in&\mathbb{R}^d\\
    \text{ATTN}(i, \bm{X})
        &:= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h} \bm{x}_j\overset{\text{new}}{\bm{W}^{VO}_h} + \overset{\text{new}}{\bm{b}^{VO}} &\hspace{1em}\in&\mathbb{R}^d\\
\end{align}
$$

where

$$
\begin{align}
    \alpha_{i, j, h} &:= \underset{\bm{x}_j,\bm{x}_j \in \bm{X}, j \leq i}{\text{softmax}}\frac{\overset{\text{new}}{\bm{q}_h}(\bm{x}_i)\overset{\text{new}}{\bm{k}_h}(\bm{x}_j)^\top}{\sqrt{d'}} &\hspace{1em}\in&\mathbb{R}\\
    \overset{\text{new}}{\bm{q}_h}(\bm{x}) &:= \bm{x}\overset{\text{new}}{\bm{W}^Q_h} + \overset{\text{new}}{\bm{b}^Q_h} &\hspace{1em}\in&\mathbb{R}^d\\
    \overset{\text{new}}{\bm{k}_h}(\bm{x}) &:= \bm{x}\overset{\text{new}}{\bm{W}^K_h} + \overset{\text{new}}{\bm{b}^K_h} &\hspace{1em}\in&\mathbb{R}^d\\
    \overset{\text{new}}{\bm{W}^Q_h} &:= \left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}\bm{W}^Q_h &\hspace{1em}\in&\mathbb{R}^{d \times d}\\
    \overset{\text{new}}{\bm{W}^K_h} &:= \left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}\bm{W}^K_h &\hspace{1em}\in&\mathbb{R}^{d \times d}\\
    \overset{\text{new}}{\bm{b}^Q_h} &:= \bm{\beta}\bm{W}^Q_h + \bm{b}^Q_h &\hspace{1em}\in&\mathbb{R}^{d}\\
    \overset{\text{new}}{\bm{b}^K_h} &:= \bm{\beta}\bm{W}^K_h + \bm{b}^K_h &\hspace{1em}\in&\mathbb{R}^{d}\\
\end{align}
$$

and

$$
\begin{align}
    \overset{\text{new}}{\bm{W}^{VO}_h} &:= \left(\bm{I}-\frac{1}{d}\bm{1}^\top\bm{1}\right)\diag{\bm{\gamma}}\bm{W}^V_h\bm{W}^O_h &\hspace{1em}\in&\mathbb{R}^{d \times d}\\
    \overset{\text{new}}{\bm{b}^{VO}} &:= \bm{\beta}\bm{W}^V\bm{W}^O + \bm{b}^V\bm{W}^O + \bm{b}^O &\hspace{1em}\in&\mathbb{R}^{d}\\
\end{align}
$$


