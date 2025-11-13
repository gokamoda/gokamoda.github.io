---
layout: splash
title: "RoPE Implementation"
use_math: true
header:
  teaser: /assets/img/rotate-half.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2025-10-03
excerpt: "How RoPE is implemented in practice."
---



## From paper
<a href="#su2020roformer">Su et al. (2020)</a>


Let $\bm{q}_m$ be the query vector at position $m$ before applying RoPE, and $\mathring{\bm{q}}_m$ be the query vector after applying RoPE.

$$
\begin{align}
	\begin{bmatrix}
		\mathring{q}_m^{(1)} \\ \mathring{q}_m^{(2)} \\ \mathring{q}_m^{(3)} \\ \mathring{q}_m^{(4)} \\
	    \vdots \\ \mathring{q}_m^{(d-1)}\\ \mathring{q}_m^{(d)}
	\end{bmatrix}
	&=
	\begin{bmatrix}
		\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
		\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
		 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0\\
		 0 & 0 & \sin m\theta_2 & \cos m\theta_2 &  \cdots & 0 & 0\\
		 \vdots & \vdots  &\vdots & \vdots& \ddots & \vdots & \vdots \\
		 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
		 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
		 
	\end{bmatrix}
	\begin{bmatrix}
		q_m^{(1)} \\ q_m^{(2)} \\ q_m^{(3)}\\ q_m^{(4)} \\ \vdots \\ q_m^{(d-1)} \\ q_m^{(d)} 
	\end{bmatrix}\\
	&=\begin{bmatrix}
		q_m^{(1)}\cos m\theta_1 - q_m^{(2)}\sin m\theta_1 \\
		q_m^{(1)}\sin m\theta_1 + q_m^{(2)}\cos m\theta_1 \\ 
		q_m^{(2)}\cos m\theta_2 - q_m^{(3)}\sin m\theta_2 \\
		q_m^{(2)}\sin m\theta_2 + q_m^{(3)}\cos m\theta_2 \\ 
		\vdots \\
		q_m^{(d-1)}\cos m\theta_{d/2} - q_m^{(d)}\sin m\theta_{d/2} \\
		q_m^{(d-1)}\sin m\theta_{d/2} + q_m^{(d)}\cos m\theta_{d/2} \\ 
	\end{bmatrix}\\
\end{align}
$$

Here, $\theta_i = \theta_\text{RoPE}^{-2(i-1)/d}$ and $\theta_\text{RoPE}$ is a hyperparameter (typically $10,000$).


## Implementation: Llama (-Llama3)

`modeling_llama.py`: [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```


In Llama implementation, the first half of the dimentions correspond to the odd dimentions in the paper. The second half correspond to the even dimentions in the paper.

$$
\begin{align}
	\begin{bmatrix}
		\mathring{q}_m^{(1)} \\ \mathring{q}_m^{(2)} \\ \vdots \\ \mathring{q}_m^{(d/2)} \\ \mathring{q}_m^{(d/2+1)} \\ \mathring{q}_m^{(d/2+2)} \\ \vdots \\ \mathring{q}_m^{(d)}
	\end{bmatrix}
	&=
	\begin{bmatrix}
		\cos m\theta_1 & 0 & 0 & 0 \\
		 0 & \cos m\theta_2 &0 & 0 & & &\bm{0} \\
		 \vdots & \vdots  &\ddots & \vdots& \\
		 0 & 0  & 0& \cos m\theta_{d/2}\\
		 &&&&\cos m\theta_1 & 0 & 0 & 0\\
		 &&&&0 & \cos m\theta_2 &0 & 0 \\
		 &\bm{0}&&&\vdots &  \vdots& \ddots & \vdots \\
		 &&&&0 & 0  & 0& \cos m\theta_{d/2}\\
	\end{bmatrix}
	\begin{bmatrix}
		q_m^{(1)} \\ q_m^{(2)} \\ \vdots \\ q_m^{(d/2)} \\ q_m^{(d/2+1)} \\ q_m^{(d/2+2)} \\ \vdots \\ q_m^{(d)}
	\end{bmatrix}\\
	&\quad + \begin{bmatrix}
		\sin m\theta_1 & 0 & 0 & 0 \\
		 0 & \sin m\theta_2 &0 & 0 & & & \bm{0} \\
		 \vdots & \vdots  &\ddots & \vdots& \\
		 0 & 0  & 0& \sin m\theta_{d/2}\\
		 &&&&\sin m\theta_1 & 0 & 0 & 0\\
		 &&&&0 & \sin m\theta_2 &0 & 0 \\
		 &\bm{0}&&&\vdots &  \vdots& \ddots & \vdots \\
		 &&&&0 & 0  & 0& \sin m\theta_{d/2}\\
	\end{bmatrix}
	\begin{bmatrix}
		- q_m^{(d/2+1)} \\ - q_m^{(d/2+2)} \\ \vdots \\ - q_m^{(d)} \\ q_m^{(1)} \\ q_m^{(2)} \\ \vdots \\ q_m^{(d/2)}
	\end{bmatrix}\\
	&=\begin{bmatrix}
		q_m^{(1)}\cos m\theta_1 - q_m^{(d/2 + 1)}\sin m\theta_1 \\
		q_m^{(2)}\cos m\theta_2 - q_m^{(d/2 + 2)}\sin m\theta_2 \\ 
		\vdots \\
		q_m^{(d/2)}\cos m\theta_{d/2} - q_m^{(d)}\sin m\theta_{d/2} \\
		q_m^{(1)}\sin m\theta_1 + q_m^{(d/2+1)}\cos m\theta_1  \\
		q_m^{(2)}\sin m\theta_2 + q_m^{(d/2+2)}\cos m\theta_2  \\
		\vdots \\
		q_m^{(d/2)}\sin m\theta_{d/2} + q_m^{(d)}\cos m\theta_{d/2}  \\
	\end{bmatrix}\\
\end{align}
$$

## Implementation: GPT-OSS

`modeling_gpt_oss.py`: [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)
```python
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)
```

In GPT-OSS implementation, the core idea is the same as Llama implementation, the size of `cos` and `sin` tensors are different. 
In Llama, $ \text{cos}, \text{sin} \in \mathbb{R}^{d} $ while in GPT-OSS, $ \text{cos}, \text{sin} \in \mathbb{R}^{d/2} $.


## References
<ol>
    <li id="su2020roformer">
        <a href="https://arxiv.org/abs/2104.09864" target="_blank" >
        Su et al. 2020, RoFormer: Enhanced Transformer with Rotary Position Embedding.
        </a>
    </li>
</ol>