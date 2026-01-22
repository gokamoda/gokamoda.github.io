---
layout: splash
title: "GPT-OSS Attnention Sink"
use_math: true
header:
  teaser: /assets/img/gptoss-bias.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2025-05-12 # moved from DissectingGPT
excerpt: "The bias term for preventing attention sink in GPT-OSS may have other effects than just preventing attention sink."
---

tl;dr: The bias term for preventing attention sink in GPT-OSS may have other effects than just preventing attention sink.
(made public on 2026-01-22)


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


## Self-Attention

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
\begin{align}
\alpha_{i, j, h} &= \frac{\exp(s_{i, j, h})}{\exp(b^{S}_h)+\sum_{j'} \exp(s_{i, j', h})}\\
s_{i, j, h} &:= \frac{\bm{q}_h(\bm{x}_i)\bm{k}_h(\bm{x}_j)^\top}{\sqrt{d'}}
\end{align}
$$

where $d' = d/H$ is the dimension of each head, and $b^{S}_h$ is a learned scalar parameter introduced in GPT-OSS for preventing attention sink.

Now, let the attention weight assigned to the sink be expressed as follows:

$$
\alpha_{i, \text{sink}, h} := \frac{\exp(b^{S}_h)}{\exp(b^{S}_h)+\sum_{j'} \exp(s_{i, j', h})}
$$

Due to the presence of $b^{S}_h$, the following holds:

$$
\begin{align}
&1 = \alpha_{i, \text{sink}, h} + \sum_j \alpha_{i, j, h}\\
&\Leftrightarrow \sum_j \alpha_{i, j, h} = 1 - \alpha_{i, \text{sink}, h}
\end{align}
$$




The output of Attention layer of an causal model at position $i$ can be expressed as follows:

$$
\begin{align}
    \text{ATTN}(i, \bm{X})
        &:=\left[\text{head}_1(i, \bm{X})\hspace{0.5em}\cdots\hspace{0.5em}\text{head}_H(i, \bm{X})\right]
            \bm{W}^O + \bm{b}^O\\
        &=\sum_{h=1}^H \text{head}_h(i, \bm{X})\bm{W}^O_h + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \bm{v}_h(\bm{x}_j)\right)\bm{W}^O_h + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \left(\bm{x}_j\bm{W}^V_h + \bm{b}^V_h\right)\right)\bm{W}^O_h + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \left(\bm{x}_j\bm{W}^V_h + \bm{b}^V_h\right)\bm{W}^O_h\right) + \bm{b}^O\\
        &=\sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h} \left(\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \bm{b}^V_h\bm{W}^O_h\right)\right) + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h}\bm{b}^V_h\bm{W}^O_h\right) + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H \left(\sum_{j=1}^i \alpha_{i, j, h}\right)\bm{b}^V_h\bm{W}^O_h + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H \left(1-\alpha_{i, \text{sink}, h}\right)\bm{b}^V_h\bm{W}^O_h + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H\bm{b}^V_h\bm{W}^O_h - \sum_{h=1}^H\alpha_{i, \text{sink}, h}\bm{b}^V_h\bm{W}^O_h + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h + \sum_{h=1}^H\alpha_{i, \text{sink}, h}(-\bm{b}^V_h\bm{W}^O_h)+ \sum_{h=1}^H\bm{b}^V_h\bm{W}^O_h  + \bm{b}^O\\
        &= \sum_{h=1}^H \sum_{j=1}^i \alpha_{i, j, h}\bm{x}_j\bm{W}^V_h\bm{W}^O_h+ \sum_{h=1}^H\alpha_{i, \text{sink}, h}(-\bm{b}^V_h\bm{W}^O_h) + \bm{b}^V\bm{W}^O + \bm{b}^O\\
\end{align}
$$

Thus, in GPT-OSS, the newly introduced attention sink bias $b^{S}_h$ 
- makes the attention weights sum to less than 1 for non-sink tokens, and
- changes the intensity of the bias term of the value transformation based on the attention weight assigned to the sink.