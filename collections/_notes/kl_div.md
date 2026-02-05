---
layout: splash
title: "torch.nn.functional.kl_div"
use_math: true
header:
  teaser: /assets/img/kl_div.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2025-12-15
excerpt: "torch.nn.functional.kl_divのメモ"
---

## メモの目的
- 以下の問いへの回答を書いておく
    - $D_{\text{KL}}(P\|\|Q)$ の、PとQはどっちが予測分布でどっちが正解分布か
    - torch.nn.functional.kl_div の引数には何を渡せばよいのか


## 一般的な式

$$
D_{KL}(P || Q) = \sum_{i} P(i) \log\frac{P(i)}{Q(i)}
$$

- `a measure of how much an approximating probability distribution Q is different from a true probability distribution P` [Wikipedia (2025-12-15)]
    - (他の参考書を参照したほうが良いのだが、まぁ。) 
- つまり、$P$ を真の分布 (正解分布)、$Q$ を予測分布とする。


## torch.nn.functional.kl_div の引数
- 結論
    - `input` は予測分布の対数を渡す
    - `target` は正解分布を渡す

- [公式ドキュメント](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) より (2025-12-15, v2.9.1):
    - `torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)`
    - `input` : Tensor of arbitrary shape in log-probabilities
    - `target` : Tensor of the same shape as input. 
    - `size_average`: Deprecated
    - `reduce`: Deprecated
    - `reduction`: Specifies the reduction to apply to the output. Default: 'mean'
        - `'none'`: no reduction will be applied
        - `'batchmean'`: the sum of the output will be divided by batchsize
        - `'sum'`: the output will be summed
        - `'mean'`: the output will be divided by the number of elements in the output

## 上の結論に至った検証コード
```python
import torch

p = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
q = torch.tensor([
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6],
])

no_reduction = torch.tensor([
    [-torch.log(torch.tensor(0.5)), 0.0, 0.0],
    [0.0, -torch.log(torch.tensor(0.3)), 0.0],
])

```

### 引数の確認とbatchmeanの動作確認
```python

def test_kl_div(p, q):
    kl_div = torch.nn.functional.kl_div(input = q.log(), target=p, reduction='batchmean')
    should_equal = no_reduction.sum(dim=1).mean(dim=0)
    assert torch.allclose(kl_div, should_equal), f"{kl_div} != {should_equal}"
test_kl_div(p, q)
```



### `reduction='none'` の動作確認
```python
def test_kl_div_none(p, q):
    kl_div = torch.nn.functional.kl_div(input = q.log(), target=p, reduction='none')
    assert torch.allclose(kl_div, no_reduction), f"{kl_div} != {no_reduction}"
test_kl_div_none(p, q)
```


### `reduction='sum'` の動作確認
```python
def test_kl_div_sum(p, q):
    kl_div = torch.nn.functional.kl_div(input = q.log(), target=p, reduction='sum')
    should_equal = no_reduction.sum(dim=1).sum(dim=0)
    assert torch.allclose(kl_div, should_equal), f"{kl_div} != {should_equal}"
test_kl_div_sum(p, q)
```


### `reduction='mean'`
WARNING: doesn’t return the true kl divergence value, please use reduction = 'batchmean' which aligns with KL math definition.

```python
def test_kl_div_mean(p, q):
    kl_div = torch.nn.functional.kl_div(input = q.log(), target=p, reduction='mean')
    should_equal = no_reduction.mean(dim=1).mean(dim=0)
    assert torch.allclose(kl_div, should_equal), f"{kl_div} != {should_equal}"
test_kl_div_mean(p, q)
```