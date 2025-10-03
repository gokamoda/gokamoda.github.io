var store = [{
        "title": "Infratop (DMM WebCamp)",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202103-infratop",
        "teaser": null
      },{
        "title": "AI王 Committee Member",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202305-aio",
        "teaser": null
      },{
        "title": "NS Solutions R&D Internship",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202309-nssol",
        "teaser": null
      },{
        "title": "AKATSUKI-SICA",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202310-akatsuki-sica",
        "teaser": null
      },{
        "title": "Hakuhodo DY Holdings Inc. Joint Research",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202310-hakuhodo",
        "teaser": null
      },{
        "title": "Visiting Student at NLP Department, MBZUAI",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202402-mbzuai",
        "teaser": null
      },{
        "title": "Tohoku University GP-DS Research Assistant",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202404-tu-gpds",
        "teaser": null
      },{
        "title": "NINJAL Part-time Researcher",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202504-ninjal-part-time-researcher",
        "teaser": null
      },{
        "title": "SOKENDAI Special Researcher Program",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202504-sokenda-srp",
        "teaser": null
      },{
        "title": "RMSNorm and LayerNorm",
        "excerpt":"Preparation   Let $\\mu(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns element-wise mean of a row-vector $\\bm{x} \\in \\mathbb{R}^{d}$:   \\[\\begin{align} \\mu(\\bm{x})  &amp;= \\frac{1}{d}\\sum_{i=1}^d \\bm{x}_i\\\\ &amp;=\\frac{1}{d}\\bm{x}\\cdot \\bm{1}\\\\ &amp;=\\frac{1}{d}\\bm{x}\\bm{1}^\\top \\end{align}\\]  Let $c(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}^{d}$ be centering function, that subtracts the element-wise mean from each element of $\\bm{x}$:   \\[\\begin{aligned} \tc(\\bm{x})&amp;=\\bm{x} - \\mu(\\bm{x})\\bm{1}\\\\ \t&amp;= \\bm{x} - \\frac{1}{d}\\bm{x}\\bm{1}^\\top\\bm{1}\\\\ \t&amp;= \\bm{x} \\left(1 - \\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right) \\end{aligned}\\]  By the way, (I - \\frac{1}{d}\\bm{1}^\\top\\bm{1}) is called the  centering matrix.   Let $\\text{RMS}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns the element-wise RMS (root mean square):   \\[\\begin{align} \t\\text{RMS}(\\bm{x})&amp;=\\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2}\\\\ \t&amp;=\\frac{\\sqrt{\\sum_{i=1}^d x_i^2}}{\\sqrt{d}}\\\\ \t&amp;=\\frac{||\\bm{x}||_2}{\\sqrt{d}} \\end{align}\\]  Let $\\text{MS}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns the squared RMS (root mean square):   \\[\\begin{align} \t\\text{MS}(\\bm{x})&amp;=\\text{RMS}(\\bm{x})^2\\\\ \t&amp;=\\frac{||\\bm{x}||_2^2}{d}\\\\ \t&amp;=\\frac{1}{d}\\sum_{i=1}^d x_i^2\\\\ \\end{align}\\]  Let $\\text{Var}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns element-wise variance:   \\[\\begin{align} \\text{Var}(\\bm{x}) &amp;= \\frac{1}{d}\\sum_{i=1}^d (x_i - \\mu(\\bm{x}))^2\\\\ &amp;=\\frac{1}{d}\\sum_{i=1}^d (\\bm{x} - \\mu(\\bm{x})\\bm{1})^2_i\\\\ &amp;=\\frac{1}{d}\\sum_{i=1}^d c(\\bm{x})_i^2\\\\ &amp;=\\text{MS}(c(\\bm{x})) \\end{align}\\]  RMSNorm  PyTorch: RMSNorm   \\[\\text{RMSNorm}(\\bm{x}) = \\frac{\\bm{x}}{\\sqrt{\\text{MS}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma}\\]  Here, $\\odot$ is element-wise multiplication, $\\bm{\\gamma}\\in \\mathbb{R}^d$ is a learnable weight vector, and $\\varepsilon$ is a small constant for numerical stability.   LayerNorm  PyTorch: LayerNorm   In the original form:   \\[\\text{LayerNorm}(\\bm{x}) = \\frac{\\bm{x} - \\mu(\\bm{x})\\bm{1}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma} +\\bm{\\beta}\\]  This can be rewritten using the centering function $c(\\bm{x})$ and the MS function $\\text{MS}(\\bm{x})$ as follows:   \\[\\begin{aligned} \\text{LayerNorm}(\\bm{x}) &amp;= \\frac{c(\\bm{x})}{\\sqrt{\\text{MS}(c(\\bm{x}))+\\varepsilon}}\\odot \\bm{\\gamma} + \\bm{\\beta}\\\\ \\end{aligned}\\]  Thus, the following holds: LayerNorm is equal to \"centering\" → RMSNorm → \"add bias\"   \\[\\text{LayerNorm}(\\bm{x}) = \\text{RMSNorm}(c(\\bm{x})) + \\bm{\\beta}\\]  Also, element-wise multiplication of $\\bm{\\gamma}$ can be expressed as matrix multiplication of $\\text{diag}(\\bm{\\gamma})$. Therefore, LayerNorm can be rewritten as:   \\[\\begin{align} \\text{LayerNorm}(\\bm{x}) &amp;= \\frac{1}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\left(\\bm{x} - \\mu(\\bm{x})\\bm{1}\\right)\\odot \\bm{\\gamma} + \\bm{\\beta}\\\\ &amp;= \\frac{\\bm{x}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\left(I - \\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\text{diag}(\\bm{\\gamma}) + \\bm{\\beta}\\\\ \\end{align}\\]  Thus only non-linear operation in LayerNorm is the division by $\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}$.  ","categories": [],
        "tags": [],
        "url": "/notes/layernorm/",
        "teaser": "/assets/img/layernorm_rmsnorm.png"
      },{
        "title": "LogitLens without bias",
        "excerpt":"   LogitLens[1] applies $\\text{LMHead}$ to the internal representations $(\\bm{h} \\in \\mathbb{R}^{1\\times d})$ of a transformer model.   \\[\\begin{equation} \\text{LMHead}(\\bm{h}) = \\text{LN}_\\text{f}(\\bm{h})\\bm{E}^O \\label{eq:lm_head} \\end{equation}\\]  Here, $\\bm{E}^O \\in \\mathbb{R}^{d\\times |\\mathcal{V}|}$ is the unembedding matrix and $\\text{LN}_\\text{f}$ is the final layer normalization of a transformer model. In this page, we assume LayerNorm (not RMSNorm) is used for $\\text{LN}_\\text{f}$ , which is defined as follows.   \\[\\begin{equation} \\text{LN}_\\text{f}(\\bm{h}) = \\frac{\\bm{x} - \\mu(\\bm{x})\\bm{1}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma} +\\bm{\\beta} \\label{eq:lm_f} \\end{equation}\\]  Here, $\\mu(\\bm{h}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ is a function that returns element-wise mean of a row-vector $\\bm{h}$ and $\\bm{\\gamma}, \\bm{\\beta} \\in \\mathbb{R}^{d}$ are learnable parameters. $\\odot$ represents element-wise multiplication.   With LogitLens, one can project the hidden states after each transformer layers to the vocabulary space.             Example of LogitLens.   By combining Equation\\eqref{eq:lm_head} and \\eqref{eq:lm_f}, we get a bias term for the projection to vocabulary space.   \\[\\begin{align} \\text{LogitLens}(\\bm{h}) &amp;= \\left(\\frac{\\bm{h} - \\mu(\\bm{h})\\bm{1}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma} + \\bm{\\beta}\\right)\\bm{E}^O\\\\ &amp;= \\left(\\frac{\\bm{h} - \\mu(\\bm{h})\\bm{1}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma}\\right)\\bm{E}^O + \\bm{\\beta}\\bm{E}^O \\label{eq:lm_head_bias} \\end{align}\\]  The second term in Equation\\eqref{eq:lm_head_bias}, which is $\\bm{\\beta}\\bm{E}^O \\in \\mathbb{R}^{|\\mathcal{V}|}$ is the bias term, which is added to the result of LogitLens regardless of the input. Adding such bias may not reasonable when analyzing “what the model’s intermediate states represent” as Kobayashi et al. (2023) reports that word frequency in the training corpus is encoded in this bias term of $\\text{LN}_\\text{f}$ in GPT-2 model.   By removing the bias term, we get the following result.             Vanilla LogitLens (GPT-2)            LogitLens w/o Bias (GPT-2)             Vanilla LogitLens (OPT)            LogitLens w/o Bias (OPT)   References                         Nostalgibraist 2020, interpreting GPT: the logit lens.                            Kobayashi et al. 2023, Transformer Language Models Handle Word Frequency in Prediction         Head.       ","categories": [],
        "tags": [],
        "url": "/notes/logitlens-wob/",
        "teaser": "/assets/img/logit_lens_nobias_gpt2.png"
      },{
        "title": "Folding weights in transformers",
        "excerpt":"This reformulation of LayerNorm and Self-Attention is used in our paper:                                   Kamoda, G.,           Heinzerling, B.,           Inaba, T.,           Kudo, K.,           Sakaguchi, K.,      &amp;           Inui, K.  (2025).  Weight-based Analysis of Detokenization in Language Models: Understanding the First Stage of Inference Without Inference.  In Findings of the Association for Computational Linguistics: NAACL 2025.                                                ACL Anthology     &nbsp;                                    arXiv     &nbsp;                                    GitHub                 Notation   \\[\\begin{alignat}{4}     &amp;\\bm{X} &amp;:= &amp;     \\begin{bmatrix}         \\bm{x}_1\\\\         \\vdots\\\\         \\bm{x}_n     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{n \\times d}\\\\     &amp;\\bm{W}^O &amp;:= &amp;     \\begin{bmatrix}         \\bm{W}^O_1\\\\         \\vdots\\\\         \\bm{W}^O_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d \\times d}\\\\     &amp;\\bm{W}^Q &amp;:= &amp;     \\begin{bmatrix}         \\bm{W}^Q_1 &amp; \\cdots &amp; \\bm{W}^Q_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d \\times d}&amp; \\label{eq:wq_split}\\\\     &amp;\\bm{W}^K &amp;:= &amp;     \\begin{bmatrix}         \\bm{W}^K_1 &amp; \\cdots &amp; \\bm{W}^K_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d \\times d}&amp; \\label{eq:wk_split}\\\\     &amp;\\bm{W}^V &amp;:= &amp;     \\begin{bmatrix}         \\bm{W}^V_1 &amp; \\cdots &amp; \\bm{W}^V_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d \\times d}&amp;\\label{eq:wv_split}\\\\     &amp;\\bm{b}^Q &amp;:= &amp;     \\begin{bmatrix}         \\bm{b}^Q_1 &amp; \\cdots &amp; \\bm{b}^Q_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d}&amp; \\label{eq:bq_split}\\\\     &amp;\\bm{b}^K &amp;:= &amp;     \\begin{bmatrix}         \\bm{b}^K_1 &amp; \\cdots &amp; \\bm{b}^K_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in&amp; \\mathbb{R}^{d}&amp; \\label{eq:bk_split}\\\\     &amp;\\bm{b}^V &amp;:= &amp;     \\begin{bmatrix}         \\bm{b}^V_1 &amp; \\cdots &amp; \\bm{b}^V_H     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d}&amp; \\\\     &amp;\\bm{I} &amp;:= &amp;     \\begin{bmatrix}         1 &amp; 0 &amp; \\cdots &amp; 0 \\\\         0 &amp; 1 &amp; \\cdots &amp; 0 \\\\         \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots \\\\         0 &amp; 0 &amp; \\cdots &amp; 1 \\\\     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d\\times d}&amp; \\\\     &amp;\\bm{1} &amp;:= &amp;     \\begin{bmatrix}         1 &amp; \\cdots &amp; 1     \\end{bmatrix}     &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d} \\end{alignat}\\]  Original LayerNorm   Layer Normalization can be expressed as follows (org stands for original):   \\[\\begin{alignat}{3}     &amp;\\text{LN}(\\bm{x}) &amp;:=&amp;\\ \\frac{\\bm{x}-\\mu(x)\\bm{1}}{\\sigma(\\bm{x})}\\odot\\bm{\\gamma} + \\bm{\\beta}&amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     &amp;\\bm{x} &amp;:=&amp;\\      \\begin{bmatrix}         x^{(1)} &amp; \\cdots &amp; x^{(d)}     \\end{bmatrix}     &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     &amp;\\mu(\\bm{x}) &amp;:=&amp;\\ \\frac{1}{d}\\sum_kx^{(k)}&amp;\\hspace{1em}\\in&amp;\\mathbb{R}\\\\     &amp;\\sigma(\\bm{x}) &amp;:=&amp;\\ \\sqrt{\\frac{1}{d}\\sum_k^d\\left(x^{(k)}-\\mu(\\bm{x})\\right)^2+\\epsilon}&amp;\\hspace{1em}\\in&amp;\\mathbb{R} \\end{alignat}\\]  Now, $\\mu(\\bm{x})$ can be reformulated as follows:   \\[\\begin{align}     \\mu(\\bm{x})\\bm{1}     &amp;=\\frac{1}{d}\\left(\\sum_kx^{(k)}\\right)\\bm{1}\\\\     &amp;=\\frac{1}{d}\\left(\\bm{x}\\bm{1}^\\top\\right)\\bm{1}\\\\     &amp;=\\bm{x}\\left(\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right) \\end{align}\\]  Thus $\\text{LN}_{\\text{org}}$ can be reformulated as follows.   \\[\\begin{align}     \\text{LN}(\\bm{x})      &amp;= \\frac{\\bm{x}-\\mu(\\bm{x})\\bm{1}}{\\sigma(\\bm{x})}\\odot\\bm{\\gamma} + \\bm{\\beta}\\\\     &amp;= \\frac{1}{\\sigma(\\bm{x})} \\left(\\bm{x}-\\bm{x}\\left(\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\right) \\diag{\\bm\\gamma}+ \\bm{\\beta}\\\\     &amp;= \\frac{\\bm{x}}{\\sigma(\\bm{x})}\\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm\\gamma}+ \\bm{\\beta} \\end{align}\\]  Original Self-Attention   Let query, key, value transformations of each head $h$ be expressed as follows:   \\[\\begin{align}     \\bm{q}_h(\\bm{x}) &amp;:= \\bm{x}\\bm{W}_h^Q + \\bm{b}_h^Q\\\\     \\bm{k}_h(\\bm{x}) &amp;:= \\bm{x}\\bm{W}_h^K + \\bm{b}_h^K\\\\     \\bm{v}_h(\\bm{x}) &amp;:= \\bm{x}\\bm{W}_h^V + \\bm{b}_h^V\\\\ \\end{align}\\]  Let attention weight from token position $i$ to $j$  ($i \\ge j$) in head $h$ be expressed as follows:   \\[\\alpha_{i, j, h} = \\underset{\\bm{x}_j,\\bm{x}_j \\in \\bm{X}, j \\leq i}{\\text{softmax}}\\frac{\\bm{q}_h(\\bm{x}_i)\\bm{k}_h(\\bm{x}_j)^\\top}{\\sqrt{d'}}\\]  where $d’ = d/H$ is the dimension of each head.   The output of Attention layer of an causal model at position $i$ can be expressed as follows:   \\[\\begin{align}     \\text{ATTN}(i, \\bm{X})         &amp;:=\\left[\\text{head}_1(i, \\bm{X})\\hspace{0.5em}\\cdots\\hspace{0.5em}\\text{head}_H(i, \\bm{X})\\right]             \\bm{W}^O + \\bm{b}^O\\\\         &amp;=\\sum_{h=1}^H \\text{head}_h(i, \\bm{X})\\bm{W}^O_h + \\bm{b}^O\\\\         &amp;=\\sum_{h=1}^H \\left(\\sum_{j=1}^i \\alpha_{i, j, h} \\bm{v}_h(\\bm{x}_j)\\right)\\bm{W}^O_h + \\bm{b}^O\\\\         &amp;=\\sum_{h=1}^H \\left(\\sum_{j=1}^i \\alpha_{i, j, h} \\left(\\bm{x}_j\\bm{W}^V_h + \\bm{b}^V_h\\right)\\right)\\bm{W}^O_h + \\bm{b}^O\\\\         &amp;= \\sum_{h=1}^H \\sum_{j=1}^i \\alpha_{i, j, h}\\bm{x}_j\\bm{W}^V_h\\bm{W}^O_h + \\sum_{h=1}^H \\left(\\sum_{j=1}^i \\alpha_{i, j, h}\\bm{b}^V_h\\right)\\bm{W}^O_h + \\bm{b}^O\\\\         &amp;= \\sum_{h=1}^H \\sum_{j=1}^i \\alpha_{i, j, h}\\bm{x}_j\\bm{W}^V_h\\bm{W}^O_h + \\sum_{h=1}^H\\bm{b}^V_h\\bm{W}^O_h + \\bm{b}^O\\hspace{0.5em} \\left(\\because \\sum_j \\alpha_{i, j, h} = 1\\right)\\\\         &amp;= \\sum_{h=1}^H \\sum_{j=1}^i \\alpha_{i, j, h}\\bm{x}_j\\bm{W}^V_h\\bm{W}^O_h + \\bm{b}^V\\bm{W}^O + \\bm{b}^O\\\\         &amp;= \\sum_{h=1}^H \\sum_{j=1}^i \\alpha_{i, j, h} \\bm{x}_j\\bm{W}^{VO}_h + \\bm{b}^{VO} \\end{align}\\]  where   \\[\\begin{align}     \\bm{W}^{VO}_h &amp;:= \\bm{W}^V_h\\bm{W}^O_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d \\times d}\\\\     \\bm{b}^{VO} &amp;:= \\bm{b}^V\\bm{W}^O + \\bm{b}^O &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d}\\\\ \\end{align}\\]  Reformulating LayerNorm and Self-Attention   LayerNorm is always followed by a linear transformation in transformers. Thus, we can fold the weights of LayerNorm into the weights of the following linear transformation.   For example, in the case of LayerNorm followed by query transformation, we can fold the weights as follows:   \\[\\begin{align}     \\bm{q}_h(\\text{LN}(\\bm{x}))         &amp;= \\text{LN}(\\bm{x})\\bm{W}^Q_h + \\bm{b}^Q_h\\\\         &amp;= \\left(\\frac{\\bm{x}}{\\sigma(\\bm{x})}\\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}+ \\bm{\\beta}\\right)\\bm{W}^Q_h + \\bm{b}^Q_h\\\\         &amp;= \\frac{\\bm{x}}{\\sigma(\\bm{x})}\\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}\\bm{W}^Q_h + \\bm{\\beta}\\bm{W}^Q_h + \\bm{b}^Q_h\\\\         &amp;= \\overset{\\text{new}}{\\text{LN}}(\\bm{x})\\overset{\\text{new}}{\\bm{W}^Q_h} + \\overset{\\text{new}}{\\bm{b}^Q_h} \\end{align}\\]  where   \\[\\begin{align}     \\overset{\\text{new}}{\\text{LN}}(\\bm{x}) &amp;:= \\frac{\\bm{x}}{\\sigma(\\bm{x})} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     \\overset{\\text{new}}{\\bm{W}^Q_h} &amp;:= \\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}\\bm{W}^Q_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d \\times d}\\\\     \\overset{\\text{new}}{\\bm{b}^Q_h} &amp;:= \\bm{\\beta}\\bm{W}^Q_h + \\bm{b}^{\\{Q, K\\}}_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d}\\\\ \\end{align}\\]  The same can be done for key and value transformations, and thus LayerNorm followed by self-attention can be reformulated as follows:   \\[\\begin{align}     \\overset{\\text{new}}{\\text{LN}}(\\bm{x}) &amp;:= \\frac{\\bm{x}}{\\sigma(\\bm{x})} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     \\overset{\\text{new}}{\\text{ATTN}}(i, \\bm{X})         &amp;:= \\sum_{h=1}^H \\sum_{j=1}^i \\alpha_{i, j, h} \\bm{x}_j\\overset{\\text{new}}{\\bm{W}^{VO}_h} + \\overset{\\text{new}}{\\bm{b}^{VO}} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\ \\end{align}\\]  where   \\[\\begin{align}     \\overset{\\text{new}}{\\alpha_{i, j, h}} &amp;:= \\underset{\\bm{x}_j,\\bm{x}_j \\in \\bm{X}, j \\leq i}{\\text{softmax}}\\frac{\\overset{\\text{new}}{\\bm{q}_h}(\\bm{x}_i)\\overset{\\text{new}}{\\bm{k}_h}(\\bm{x}_j)^\\top}{\\sqrt{d'}} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}\\\\     \\overset{\\text{new}}{\\bm{q}_h}(\\bm{x}) &amp;:= \\bm{x}\\overset{\\text{new}}{\\bm{W}^Q_h} + \\overset{\\text{new}}{\\bm{b}^Q_h} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     \\overset{\\text{new}}{\\bm{k}_h}(\\bm{x}) &amp;:= \\bm{x}\\overset{\\text{new}}{\\bm{W}^K_h} + \\overset{\\text{new}}{\\bm{b}^K_h} &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^d\\\\     \\overset{\\text{new}}{\\bm{W}^Q_h} &amp;:= \\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}\\bm{W}^Q_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d \\times d}\\\\     \\overset{\\text{new}}{\\bm{W}^K_h} &amp;:= \\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}\\bm{W}^K_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d \\times d}\\\\     \\overset{\\text{new}}{\\bm{b}^Q_h} &amp;:= \\bm{\\beta}\\bm{W}^Q_h + \\bm{b}^Q_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d}\\\\     \\overset{\\text{new}}{\\bm{b}^K_h} &amp;:= \\bm{\\beta}\\bm{W}^K_h + \\bm{b}^K_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d}\\\\ \\end{align}\\]  and   \\[\\begin{align}     \\overset{\\text{new}}{\\bm{W}^{VO}_h} &amp;:= \\left(\\bm{I}-\\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\diag{\\bm{\\gamma}}\\bm{W}^V_h\\bm{W}^O_h &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d \\times d}\\\\     \\overset{\\text{new}}{\\bm{b}^{VO}} &amp;:= \\bm{\\beta}\\bm{W}^V\\bm{W}^O + \\bm{b}^V\\bm{W}^O + \\bm{b}^O &amp;\\hspace{1em}\\in&amp;\\mathbb{R}^{d}\\\\ \\end{align}\\]  ","categories": [],
        "tags": [],
        "url": "/notes/fold-weights/",
        "teaser": "/assets/img/folding-weights.png"
      },{
        "title": "メモ: PCA",
        "excerpt":"$n$ 個の $d$ 次元のデータ $\\bm{X}$ を、$d’$ 次元に圧縮するための線形変換行列 $\\bm{W}$ を求める。   \\[\\bm{X} =\\begin{bmatrix} \\bm{x}_1 \\\\ \\bm{x}_2 \\\\ \\vdots \\\\ \\bm{x}_n \\end{bmatrix} = \\begin{bmatrix} x_{1,1} &amp; x_{1,2} &amp; \\cdots &amp; x_{1,d} \\\\ x_{2,1} &amp; x_{2,2} &amp; \\cdots &amp; x_{2,d} \\\\ \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots \\\\ x_{n,1} &amp; x_{n,2} &amp; \\cdots &amp; x_{n,d} \\end{bmatrix} \\in \\mathbb{R}^{n \\times d}\\]  \\[\\bm{W} = \\begin{bmatrix} \\bm{w}_1 &amp; \\bm{w}_2 &amp; \\cdots &amp; \\bm{w}_{d'} \\\\ \\end{bmatrix} = \\begin{bmatrix} w_{1,1} &amp; w_{1,2} &amp; \\cdots &amp; w_{1,d'} \\\\ w_{2,1} &amp; w_{2,2} &amp; \\cdots &amp; w_{2,d'} \\\\ \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots \\\\ w_{d,1} &amp; w_{d,2} &amp; \\cdots &amp; w_{d,d'} \\end{bmatrix} \\in \\mathbb{R}^{d \\times d'}\\]  変換後のデータ $\\bm{Y}$ は次のように表される。 \\(\\bm{Y} = \\bm{X} \\bm{W} \\in \\mathbb{R}^{n \\times d'}\\)   PCAでは、圧縮後の各次元の分散を最大化することを考える。  圧縮後の $j\\ (1\\le j \\le d’)$次元目 の分散は、定義より以下で表せる:   \\[\\begin{align} s_j^2  &amp;= \\frac{1}{n}\\sum_{i=1}^{n}(y_{ij}-\\bar{y_j})^2\\\\ &amp;= \\frac{1}{n}\\sum_{i=1}^{n}\\left(     \\bm{x}_i \\bm{w}_j - \\frac{1}{n}\\sum_{k=1}^{n}\\bm{x}_k \\bm{w}_j \\right)^2 \\\\ &amp;= \\frac{1}{n}\\sum_{i=1}^{n}\\left(     \\left(\\bm{x}_i - \\frac{1}{n}\\sum_{k=1}^{n}\\bm{x}_k\\right) \\bm{w}_j \\right)^2 \\\\ \\end{align}\\]  ただし $\\bar{y_j} = \\frac{1}{n}\\sum_{i=1}^{n}y_{ij}$ は $j$ 次元目の平均値を表す。   ここで、$\\bar{\\bm{x}} = \\frac{1}{n}\\sum_{i=1}^{n}\\bm{x}_i \\in \\mathbb{R}^{d}$ とし、 中心化 (各次元の平均をゼロに) されたデータを $\\bm{X}^c$ とする。 $\\bm{X}^c$ は以下で定義される。   \\[\\begin{align} \\bm{X}^c &amp;= \\begin{bmatrix} \\bm{x}_1 \\\\ \\bm{x}_2 \\\\ \\vdots \\\\ \\bm{x}_n \\end{bmatrix} -  \\begin{bmatrix} \\bar{\\bm{x}} \\\\ \\bar{\\bm{x}} \\\\ \\vdots \\\\ \\bar{\\bm{x}} \\\\ \\end{bmatrix} =\\begin{bmatrix} \\bm{x}_1^c \\\\ \\bm{x}_2^c \\\\ \\vdots \\\\ \\bm{x}_n^c \\end{bmatrix} \\end{align}\\]  \\[\\begin{align} \\bm{x}^c_i &amp;=\\bm{x}_i - \\bar{\\bm{x}}\\\\ &amp;=\\bm{x}_i - \\frac{1}{n}\\sum_{k=1}^{n}\\bm{x}_k \\end{align}\\]  すると、次元 $j\\ (1\\le j \\le d’)$ の分散は以下で表せる:   \\[\\begin{align} s_j^2  &amp;= \\frac{1}{n}\\sum_{i=1}^{n}(\\bm{x}_i^c \\bm{w}_j)^2 \\\\ &amp;= \\frac{1}{n}\\sum_{i=1}^{n}(\\bm{x}_i\\bm{w}_j^c)^\\top (\\bm{x}_i^c \\bm{w}_j) \\\\ &amp;= \\frac{1}{n}\\sum_{i=1}^{n}\\bm{w}_j^\\top(\\bm{x}_i^c\\top\\bm{x}_i^c)\\bm{w}_j \\\\ &amp;= \\frac{1}{n}\\bm{w}_j^\\top \\left(\\sum_{i=1}^{n}\\bm{x}_i^c\\top\\bm{x}_i^c\\right) \\bm{w}_j \\\\ &amp;= \\bm{w}_j^\\top \\left(\\frac{1}{n}\\bm{X}^{c\\top}\\bm{X}^c\\right) \\bm{w}_j \\\\ &amp;= \\bm{w}_j^\\top \\bm{S} \\bm{w}_j \\end{align}\\]  ここで、$\\bm{S} \\in \\mathbb{R}^{d \\times d}$ はデータ$\\bm{X}$の、共分散行列と呼ばれる。   PCAでは、$s_j^2 (1\\le j \\le d’)$を最大化する $\\bm{w}_j$ を求める。 なお、ここで$\\bm{w}_j$を大きくすれば$s_j^2$も大きくなるため、$\\bm{w}_j$は単位ベクトル、つまり \\(\\|\\bm{w}_j\\|^2 = \\bm{w}_j^\\top \\bm{w}_j = 1\\) の成約を設ける。   結果、解きたい問題は   \\[\\underset{\\bm{w}_j}{\\text{argmax}}\\quad \\bm{w}_j^\\top \\bm{S} \\bm{w}_j \\quad \\text{subject to}\\quad  \\bm{w}_j^\\top \\bm{w}_j = 1\\]  これをラグランジュの未定乗数法を用いて解くことを考える。ラグランジュ関数は以下のようになる。   \\[\\begin{align} \\mathcal{L}(\\bm{w}_j, \\lambda) &amp;= \\bm{w}_j^\\top \\bm{S} \\bm{w}_j - \\lambda (\\bm{w}_j^\\top \\bm{w}_j - 1) \\\\ \\end{align}\\]  これを$W_j$で微分すれば、   \\[\\begin{align} \\frac{\\partial \\mathcal{L}}{\\partial \\bm{w}_j} &amp;= 2 \\bm{S} \\bm{w}_j - 2 \\lambda \\bm{w}_j \\end{align}\\]  したがって、以下を満たすときに極値をとる。   \\[\\bm{S} \\bm{w}_j = \\lambda \\bm{w}_j\\]  したがって、$\\bm{w}_j$は$\\bm{S}$の固有ベクトルであり、$\\lambda$は対応する固有値である。  また、$s_j^2$は、以下のように書き換えられる。 \\(\\begin{align} s_j^2 &amp;= \\bm{w}_j^\\top \\bm{S} \\bm{w}_j \\\\ &amp;= \\bm{w}_j^\\top \\lambda \\bm{w}_j \\\\ &amp;= \\lambda \\bm{w}_j^\\top \\bm{w}_j \\\\ &amp;= \\lambda \\end{align}\\)   PCAでは、$d’$ 個の次元の分散を最大化することを考えるため、$\\bm{S}$の固有値が大きい順に対応する$d’$ 個の固有ベクトルをとり並べたものを$\\bm{W}$とすればよい。   (復習) 固有値分解   行列 $\\bm{A} \\in \\mathbb{R}^{d\\times d}$ の固有値分解は、直交行列$\\bm{V}$を用いて以下のように表される。   \\[\\bm{A} = \\bm{V} \\bm{\\Lambda} \\bm{V}^\\top\\]  ここで、   \\[\\bm{\\Lambda} = \\text{diag}(\\lambda_1, \\lambda_2, \\ldots, \\lambda_d)\\]  \\[\\bm{V} = \\begin{bmatrix} \\bm{v}_1 &amp; \\bm{v}_2 &amp; \\cdots &amp; \\bm{v}_d \\end{bmatrix}\\]  とする。 $\\lambda_i$ は $\\bm{A}$ の固有値、$\\bm{v}_i$ は対応する固有ベクトルである。   また、以下のように変形できる。   \\[\\begin{align} \\bm{A} &amp;= \\bm{V} \\bm{\\Lambda} \\bm{V}^\\top\\\\ &amp;=  \\begin{bmatrix} \\bm{v}_1 &amp; \\bm{v}_2 &amp; \\cdots &amp; \\bm{v}_d \\end{bmatrix} \\begin{bmatrix} \\lambda_1 \\bm{v}_1 \\\\ \\lambda_2 \\bm{v}_2 \\\\ \\vdots \\\\ \\lambda_d \\bm{v}_d \\end{bmatrix} \\\\ &amp;= \\sum_{i=1}^{d} \\lambda_i \\bm{v}_i \\bm{v}_i^\\top \\end{align}\\]  $\\bm{S}$の固有値/ベクトルは$X$の特異値分解 (SVD) で求められる   再掲: $\\bm{S} \\in \\mathbb{R}^{d\\times d}$, $\\bm{X}^c \\in \\mathbb{R}^{n\\times d}$.   変換行列 $\\bm{W} \\in \\mathbb{R}^{d\\times d’}$ を求めるために、$\\bm{S}$の固有値を求める必要がある。 ここで、$\\bm{S}$ を $\\bm{X}^c$ に関して書き換え、$\\bm{X}^c$ の特異値分解 (SVD) を用いると以下のようになる。   \\[\\begin{align} \\bm{S} &amp;= \\frac{1}{n} \\bm{X}^{c\\top} \\bm{X}^c \\\\ &amp;= \\frac{1}{n} \\left(\\bm{U}\\bm{\\Sigma} \\bm{V}^{\\top}\\right)^\\top \\left(\\bm{U}\\bm{\\Sigma} \\bm{V}^{\\top}\\right) \\\\ &amp;= \\frac{1}{n} \\bm{V} \\bm{\\Sigma}^\\top \\bm{U}^\\top \\bm{U} \\bm{\\Sigma} \\bm{V}^{\\top} \\\\ &amp;= \\frac{1}{n} \\bm{V} \\bm{\\Sigma}^{2} \\bm{V}^{\\top}\\quad (\\because \\bm{U} \\text{ is orthogonal})\\\\ &amp;= \\bm{V} \\left(\\frac{\\bm{\\Sigma}^{2}}{n} \\right) \\bm{V}^{\\top} \\\\ \\end{align}\\]  つまり、以下が言える:     $\\bm{S}$ の 固有値は [{$\\bm{X}^c$ を特異値分解して得られる固有値からなる行列$\\bm{\\Sigma}$} を用いて表される $\\bm{\\Sigma}^{2}/n$ ]の各対角成分   $\\bm{S}$ の 固有ベクトルは $\\bm{X}^c$ を特異値分解して得られる $\\bm{V}$ の各列ベクトルである。   したがって、PCAを行う際は、$\\bm{X}^{c\\top}\\bm{X}^c$ を計算する必要はなく、$\\bm{X}^c$ を特異値分解して、$\\bm{V}$ のうち、対応する固有値が大きい $d’$ 個の列ベクトルをとり並べたものを $\\bm{W}$ とすればよい。  ","categories": [],
        "tags": [],
        "url": "/notes/pca/",
        "teaser": "/assets/img/pca.png"
      },{
        "title": "✅ Paper accepted to the COLING 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to COLING 2025.                                    Kamoda, G.,           Asai, A.,           Brassard, A.,      &amp;           Sakaguchi, K.  (2025).  Quantifying the Influence of Evaluation Aspects on Long-Form Response Assessment.  In Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025).                                                ACL Anthology     &nbsp;                                    GitHub                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/coling2025-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the ICLR 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to ICLR 2025.                               Deguchi, H.,                Kamoda, G.,           Matsushita, Y.,           Taguchi, C.,           Waga, M.,           Suenaga, K.,      &amp;           Yokoi, S.  (2025).  SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches.  In The Thirteenth International Conference on Learning Representations (ICLR 2025).                                                OpenReview     &nbsp;                                    arXiv     &nbsp;                                    Project Page                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/iclr2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentations at NLP 2025",
        "excerpt":"言語処理学会 第31回年次大会 (NLP 2025) にて、以下の6件の発表があります。                                  出口祥之,                鴨田豪,           松下祐介,           田口智大,           末永幸平,           和賀正樹,           横井祥  (2024).  SoftMatcha: 大規模コーパス検索のための柔らかくも高速なパターンマッチャー.  言語処理学会 第31回年次大会, pp. 3310-3315.                                                 予稿                                       鴨田豪,           Benjamin Heinzerling,           稲葉達郎,           工藤慧音,           坂口慶祐,           乾健太郎  (2025).  言語モデルのパラメータから探るDetokenizationメカニズム.  言語処理学会 第31回年次大会, pp. 634-639.                                                 予稿                                  小林春斗,           原知正,                鴨田豪,           横井祥  (2025).  層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ.  言語処理学会 第31回年次大会, pp. 1751-1756.                                                 予稿                                  工藤慧音,                鴨田豪,           塩野大輝,           鈴木潤  (2025).  日本語バイト符号化マスク言語モデルの開発と分析.  言語処理学会 第31回年次大会, pp. 3356-3361.                                                 予稿     &nbsp;                                    ByBERT-JP     &nbsp;                                    ByGPT-JP                                  佐々木睦史,           高橋良允,                鴨田豪,           Benjamin Heinzerling,           坂口慶祐,           乾健太郎  (2025).  LM は日本の時系列構造をどうエンコードするか.  言語処理学会 第31回年次大会, pp. 2642-2647.                                                 予稿                                  佐藤宏亮,                鴨田豪,           Benjamin Heinzerling,           坂口慶祐  (2025).  言語モデルの内部表現における文法情報の局所性について.  言語処理学会 第31回年次大会, pp. 697-701.                                                 予稿                      -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/nlp-presentation/",
        "teaser": null
      },{
        "title": "👑 2 papers received awards at NLP 2025",
        "excerpt":"言語処理学会 第31回年次大会 (NLP 2025) にて、以下の2件の論文がそれぞれ「若手奨励賞」「日本経済新聞社 CDIO室賞」を受賞しました。                               小林春斗,           原知正,                鴨田豪,           横井祥  (2025).  層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ.  言語処理学会 第31回年次大会, pp. 1751-1756.                 若手奨励賞 (20/487)                                                 予稿                                    佐々木睦史,           高橋良允,                鴨田豪,           Benjamin Heinzerling,           坂口慶祐,           乾健太郎  (2025).  LM は日本の時系列構造をどうエンコードするか.  言語処理学会 第31回年次大会, pp. 2642-2647.                 日本経済新聞社 CDIO室賞                                                 予稿                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/nlp-awards/",
        "teaser": null
      },{
        "title": "👑 Completed Master's course and received the Dean's Award",
        "excerpt":"東北大学 情報科学研究科 学位授与式にて、学位記 (修士: 情報科学) とともに「研究科長賞」をいただきました。  ","categories": ["News"],
        "tags": [],
        "url": "/news/dean-award/",
        "teaser": null
      },{
        "title": "🌸 Enrollment in SOKENDAI for Doctoral Program",
        "excerpt":"総合研究大学院大学 (SOKENDAI) 日本語言語科学コースの博士後期課程に進学しました。  JST BOOST の支援を受けるSOKENDAI特別研究員として、国立国語研究所 (NINJAL) で研究活動を行います。  ","categories": ["News"],
        "tags": [],
        "url": "/news/sokendai-enrollment/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the Findings of EMNLP 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to the Findings of EMNLP 2025.                               Inaba, T.,                Kamoda, G.,           Inui, K.,           Isonuma, M.,           Miyao, Y.,           Oseki, Y.,           Takagi, Y.,      &amp;           Heinzerling, B.  (2025).  How a Bilingual LM Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders.  In Findings of the Association for Computational Linguistics: EMNLP 2025.                                                OpenReview                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/emnlp2025findings-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the 8th BlackboxNLP Workshop",
        "excerpt":"We are pleased to announce that our paper has been accepted to the 8th BlackboxNLP Workshop.                               Takahashi, R.,                Kamoda, G.,           Heinzerling, B.,           Sakaguchi, K.,      &amp;           Inui, K.  (2025).  Understanding the Side Effects of Rank-One Knowledge Editing.  In BlackboxNLP 2025: The 8th Workshop on Analyzing and Interpreting Neural Networks for NLP.                                                arXiv                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/bbnlp2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentation at YANS 2025",
        "excerpt":"YANS 2025 にて以下の発表があります。                                    鴨田豪,           熊谷雄介,           松井孝太,           横井祥  (2025).  密度比の直接推定に基づく言語モデルの出力較正.  第20回言語処理若手シンポジウム (YANS).                   -&gt; Publications  ","categories": ["Blog"],
        "tags": [],
        "url": "/blog/yans2025-presentation/",
        "teaser": null
      },{
        "title": "Test-time Augmentation for Factual Probing",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202303-nlp-kamoda",
        "teaser": null
      },{
        "title": "長文生成の多面的評価:人手評価と自動評価の向上を目指して",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202403-nlp-kamoda",
        "teaser": null
      },{
        "title": "言語モデルからの知識削除：頻出実体の知識は副作用が破滅的",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202403-nlp-takahashi",
        "teaser": null
      },{
        "title": "大規模言語モデルの情報推薦バイアスの較正",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202405-jsai-kumagae",
        "teaser": null
      },{
        "title": "柔らかいgrep/KWICに向けて：高速単語列マッチングの埋め込み表現による連続化",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202409-yans-deguchi",
        "teaser": null
      },{
        "title": "事前学習–文脈内学習パラダイムで生じる頻度バイアスの較正",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202409-yans-ito",
        "teaser": null
      },{
        "title": "層同士の接続可能性と各層が影響を与える部分空間の重なり度合いの関係性",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202409-yans-kobayashi",
        "teaser": null
      },{
        "title": "SoftMatcha: 大規模コーパス検索のための柔らかくも高速なパターンマッチャー",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-deguchi",
        "teaser": null
      },{
        "title": "言語モデルのパラメータから探るDetokenizationメカニズム",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-kamoda",
        "teaser": null
      },{
        "title": "層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-kobayashi",
        "teaser": null
      },{
        "title": "日本語バイト符号化マスク言語モデルの開発と分析",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-kudo",
        "teaser": null
      },{
        "title": "LM は日本の時系列構造をどうエンコードするか",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-sasaki",
        "teaser": null
      },{
        "title": "言語モデルの内部表現における文法情報の局所性について",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202503-nlp-satoh",
        "teaser": null
      },{
        "title": "密度比の直接推定に基づく言語モデルの出力較正",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202509-yans-kamoda",
        "teaser": null
      },{
        "title": "Test-time Augmentation for Factual Probing",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202311-emnlp-kamoda",
        "teaser": null
      },{
        "title": "Quantifying the Influence of Evaluation Aspects on Long-Form Response Assessment",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202501-coling-kamoda",
        "teaser": null
      },{
        "title": "SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202504-iclr-deguchi",
        "teaser": null
      },{
        "title": "Weight-based Analysis of Detokenization in Language Models: Understanding the First Stage of Inference Without Inference",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202504-naacl-kamoda",
        "teaser": null
      },{
        "title": "Understanding the Side Effects of Rank-One Knowledge Editing",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202511-blackboxnlp-takahashi",
        "teaser": null
      },{
        "title": "How a Bilingual LM Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202511-emnlp-inaba",
        "teaser": null
      },{
        "title": "Can Language Models Handle a Non-Gregorian Calendar?",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202311-arxiv-sasaki",
        "teaser": null
      },{
    "title": "Notes",
    "excerpt":"","url": "/notes/"
  },{
    "title": "Publications & Activities",
    "excerpt":"## Education - Doctoral Student [2025.04 - ] Graduate Institute for Advanced Studies, SOKENDAI.Supervisor: Assoc. Prof. Sho Yokoi - Master of Information Science [2023.04 - 2025.03]Graduate School of Information Sciences, Tohoku University.Supervisor: Prof. Jun Suzuki & Assoc. Prof. Keisuke Sakaguchi Dean Award (4/126) - Bachelor of Engineering [2020.04 - 2023.03]School of Engineering, Tohoku University.Supervisor: Prof. Kentaro Inui & Assoc. Prof. Keisuke SakaguchiEarly Graduation (1/252)  ## International Conferences          {% for publication in site.pubInternationalConferences reversed %}              {% include pub-apa-international-conf.html  %}            {% endfor %}           ## Domestic Conferences          {% for publication in site.pubDomesticConferences reversed %}              {% include pub-anlp-domestic-conf.html  %}            {% endfor %}        {% if site.pubPreprint.size > 0 %} ## Preprints         {% for publication in site.pubPreprint reversed %}              {% include pub-apa-international-conf.html  %}            {% endfor %}      {% endif %}  {% if site.experiences.size > 0 %} ## Experiences         {% for experience in site.experiences reversed %}              {% include experiences.html  %}            {% endfor %}      {% endif %}           {% for talk in site.oubInvitedTalks reversed %}                       {% for speaker in talk.speakers %}           {{ speaker.name }}           {%- if forloop.last == false -%}             ,           {% endif %}         {%- endfor -%}         .         {{ talk.title }}.         {{ talk.event_name }},         {{ talk.month }}         {{ talk.year }}.         {% if talk.links %}           [           {%- for link in talk.links -%}             {{ link.name}}{% if forloop.last == false %}, {% endif %}           {%- endfor -%}           ]         {% endif %}            {% endfor %}         ","url": "/cv/"
  },{
    "title": "",
    "excerpt":"","url": "/index.html"
  },{
    "title": " - page 2",
    "excerpt":"","url": "/page/2/index.html"
  }]
