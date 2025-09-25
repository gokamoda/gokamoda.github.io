var store = [{
        "title": "RMSNorm and LayerNorm",
        "excerpt":"Preparation   Let $\\mu(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns element-wise mean of a row-vector $\\bm{x} \\in \\mathbb{R}^{d}$:   \\[\\begin{align} \\mu(\\bm{x})  &amp;= \\frac{1}{d}\\sum_{i=1}^d \\bm{x}_i\\\\ &amp;=\\frac{1}{d}\\bm{x}\\cdot \\bm{1}\\\\ &amp;=\\frac{1}{d}\\bm{x}\\bm{1}^\\top \\end{align}\\]  Let $c(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}^{d}$ be centering function, that subtracts the element-wise mean from each element of $\\bm{x}$:   \\[\\begin{aligned} \tc(\\bm{x})&amp;=\\bm{x} - \\mu(\\bm{x})\\bm{1}\\\\ \t&amp;= \\bm{x} - \\frac{1}{d}\\bm{x}\\bm{1}^\\top\\bm{1}\\\\ \t&amp;= \\bm{x} \\left(1 - \\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right) \\end{aligned}\\]  By the way, (I - \\frac{1}{d}\\bm{1}^\\top\\bm{1}) is called the  centering matrix.   Let $\\text{RMS}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns the element-wise RMS (root mean square):   \\[\\begin{align} \t\\text{RMS}(\\bm{x})&amp;=\\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2}\\\\ \t&amp;=\\frac{\\sqrt{\\sum_{i=1}^d x_i^2}}{\\sqrt{d}}\\\\ \t&amp;=\\frac{||\\bm{x}||_2}{\\sqrt{d}} \\end{align}\\]  Let $\\text{MS}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns the squared RMS (root mean square):   \\[\\begin{align} \t\\text{MS}(\\bm{x})&amp;=\\text{RMS}(\\bm{x})^2\\\\ \t&amp;=\\frac{||\\bm{x}||_2^2}{d}\\\\ \t&amp;=\\frac{1}{d}\\sum_{i=1}^d x_i^2\\\\ \\end{align}\\]  Let $\\text{Var}(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns element-wise variance:   \\[\\begin{align} \\text{Var}(\\bm{x}) &amp;= \\frac{1}{d}\\sum_{i=1}^d (x_i - \\mu(\\bm{x}))^2\\\\ &amp;=\\frac{1}{d}\\sum_{i=1}^d (\\bm{x} - \\mu(\\bm{x})\\bm{1})^2_i\\\\ &amp;=\\frac{1}{d}\\sum_{i=1}^d c(\\bm{x})_i^2\\\\ &amp;=\\text{MS}(c(\\bm{x})) \\end{align}\\]  RMSNorm  PyTorch: RMSNorm   \\[\\text{RMSNorm}(\\bm{x}) = \\frac{\\bm{x}}{\\sqrt{\\text{MS}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma}\\]  Here, $\\odot$ is element-wise multiplication, $\\bm{\\gamma}\\in \\mathbb{R}^d$ is a learnable weight vector, and $\\varepsilon$ is a small constant for numerical stability.   LayerNorm  PyTorch: LayerNorm   In the original form:   \\[\\text{LayerNorm}(\\bm{x}) = \\frac{\\bm{x} - \\mu(\\bm{x})\\bm{1}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\odot \\bm{\\gamma} +\\bm{\\beta}\\]  This can be rewritten using the centering function $c(\\bm{x})$ and the MS function $\\text{MS}(\\bm{x})$ as follows:   \\[\\begin{aligned} \\text{LayerNorm}(\\bm{x}) &amp;= \\frac{c(\\bm{x})}{\\sqrt{\\text{MS}(c(\\bm{x}))+\\varepsilon}}\\odot \\bm{\\gamma} + \\bm{\\beta}\\\\ \\end{aligned}\\]  Thus, the following holds: LayerNorm is equal to \"centering\" → RMSNorm → \"add bias\"   \\[\\text{LayerNorm}(\\bm{x}) = \\text{RMSNorm}(c(\\bm{x})) + \\bm{\\beta}\\]  Also, element-wise multiplication of $\\bm{\\gamma}$ can be expressed as matrix multiplication of $\\text{diag}(\\bm{\\gamma})$. Therefore, LayerNorm can be rewritten as:   \\[\\begin{align} \\text{LayerNorm}(\\bm{x}) &amp;= \\frac{1}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\left(\\bm{x} - \\mu(\\bm{x})\\bm{1}\\right)\\odot \\bm{\\gamma} + \\bm{\\beta}\\\\ &amp;= \\frac{\\bm{x}}{\\sqrt{\\text{Var}(\\bm{x})+\\varepsilon}}\\left(I - \\frac{1}{d}\\bm{1}^\\top\\bm{1}\\right)\\text{diag}(\\bm{\\gamma}) + \\bm{\\beta}\\\\ \\end{align}\\]  Thus only non-linear operation in LayerNorm is the division by $s(\\bm{x})$.   ","categories": [],
        "tags": [],
        "url": "/notes/sample/",
        "teaser": "/assets/img/centering_matrix.png"
      },{
        "title": "✅ Paper accepted to the COLING 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to COLING 2025.                                    Kamoda, G.     ,                Asai, A.     ,                Brassard, A.     ,      &amp;                Sakaguchi, K.       (2025).  Quantifying the Influence of Evaluation Aspects on Long-Form Response Assessment.  In Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025).       [ACL Anthology, GitHub]            -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/coling2025-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the ICLR 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to ICLR 2025.                                    Deguchi, H.     ,                Kamoda, G.     ,                Matsushita, Y.     ,                Taguchi, C.     ,                Waga, M.     ,                Suenaga, K.     ,      &amp;                Yokoi, S.       (2025).  SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches.  In The Thirteenth International Conference on Learning Representations (ICLR 2025).       [OpenReview, arXiv, Project Page]            -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/iclr2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentations at NLP 2025",
        "excerpt":"言語処理学会 第31回年次大会 (NLP 2025) にて、以下の6件の発表があります。                                       出口祥之     ,                鴨田豪     ,                松下祐介     ,                田口智大     ,                末永幸平     ,                和賀正樹     ,                横井祥       (2024).  SoftMatcha: 大規模コーパス検索のための柔らかくも高速なパターンマッチャー.  言語処理学会 第31回年次大会, pp. 3310-3315.        [予稿]                                  鴨田豪     ,                Benjamin Heinzerling     ,                稲葉達郎     ,                工藤慧音     ,                坂口慶祐     ,                乾健太郎       (2025).  言語モデルのパラメータから探るDetokenizationメカニズム.  言語処理学会 第31回年次大会, pp. 634-639.        [予稿]                                  小林春斗     ,                原知正     ,                鴨田豪     ,                横井祥       (2025).  層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ.  言語処理学会 第31回年次大会, pp. 1751-1756.        [予稿]                                  工藤慧音     ,                鴨田豪     ,                塩野大輝     ,                鈴木潤       (2025).  日本語バイト符号化マスク言語モデルの開発と分析.  言語処理学会 第31回年次大会, pp. 3356-3361.        [予稿, 🤗ByBERT-JP, 🤗ByGPT-JP]                                  佐々木睦史     ,                高橋良允     ,                鴨田豪     ,                Benjamin Heinzerling     ,                坂口慶祐     ,                乾健太郎       (2025).  LM は日本の時系列構造をどうエンコードするか.  言語処理学会 第31回年次大会, pp. 2642-2647.        [予稿]                                  佐藤宏亮     ,                鴨田豪     ,                Benjamin Heinzerling     ,                坂口慶祐       (2025).  言語モデルの内部表現における文法情報の局所性について.  言語処理学会 第31回年次大会, pp. 697-701.        [予稿]                 -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/nlp-presentation/",
        "teaser": null
      },{
        "title": "👑 2 papers received awards at NLP 2025",
        "excerpt":"言語処理学会 第31回年次大会 (NLP 2025) にて、以下の2件の論文がそれぞれ「若手奨励賞」「日本経済新聞社 CDIO室賞」を受賞しました。                                    小林春斗     ,                原知正     ,                鴨田豪     ,                横井祥       (2025).  層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ.  言語処理学会 第31回年次大会, pp. 1751-1756.                 若手奨励賞 (20/487)        [予稿]                                    佐々木睦史     ,                高橋良允     ,                鴨田豪     ,                Benjamin Heinzerling     ,                坂口慶祐     ,                乾健太郎       (2025).  LM は日本の時系列構造をどうエンコードするか.  言語処理学会 第31回年次大会, pp. 2642-2647.                 日本経済新聞社 CDIO室賞        [予稿]            -&gt; Publications  ","categories": ["News"],
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
        "excerpt":"We are pleased to announce that our paper has been accepted to the Findings of EMNLP 2025.                                    Inaba, T.     ,                Kamoda, G.     ,                Inui, K.     ,                Isonuma, M.     ,                Miyao, Y.     ,                Oseki, Y.     ,                Takagi, Y.     ,      &amp;                Heinzerling, B.       (2025).  How a Bilingual LM Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders.  In Findings of the Association for Computational Linguistics: EMNLP 2025.       [OpenReview]            -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/emnlp2025findings-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the 8th BlackboxNLP Workshop",
        "excerpt":"We are pleased to announce that our paper has been accepted to the 8th BlackboxNLP Workshop.                                    Takahashi, R.     ,                Kamoda, G.     ,                Heinzerling, B.     ,                Sakaguchi, K.     ,      &amp;                Inui, K.       (2025).  Understanding the Side Effects of Rank-One Knowledge Editing.  In The 8th BlackboxNLP Workshop.       [arXiv]            -&gt; Publications  ","categories": ["News"],
        "tags": [],
        "url": "/news/bbnlp2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentation at YANS 2025",
        "excerpt":"YANS 2025 にて以下の発表があります。                                    鴨田豪     ,                熊谷雄介     ,                松井孝太     ,                横井祥       (2024).  密度比の直接推定に基づく言語モデルの出力較正.  第20回言語処理若手シンポジウム (YANS).                   -&gt; Publications  ","categories": ["Blog"],
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
    "excerpt":"## Education - Ph.D. in Philosophy, April 2025 - March 2028 (expected).Graduate Institute for Advanced Studies, SOKENDAI.Supervisor: Assoc. Prof. Sho Yokoi - Master of Information Science, April 2023 - March 2025.Graduate School of Information Sciences, Tohoku University.Supervisor: Prof. Jun Suzuki & Assoc. Prof. Keisuke Sakaguchi Dean Award (4/126) - Bachelor of Engineering, April 2020 - March 2023.School of Engineering, Tohoku University.Supervisor: Prof. Kentaro Inui & Assoc. Prof. Keisuke SakaguchiEarly Graduation (1/252)  ## International Conferences          {% for publication in site.pubInternationalConferences reversed %}              {% include pub-apa-international-conf.html  %}            {% endfor %}           ## Domestic Conferences          {% for publication in site.pubDomesticConferences reversed %}              {% include pub-anlp-domestic-conf.html  %}            {% endfor %}        {% if site.pubPreprint.size > 0 %} ## Preprints         {% for publication in site.pubPreprint reversed %}              {% include pub-apa-international-conf.html  %}            {% endfor %}      {% endif %}   ## Experiences - [2025.04 -] SOKENDAISpecial Researcher Program (Supported by JST BOOST) - [2025.04 -] NINJALPart-time Researcher - [2023.10 -] Joint Research with Hakuhodo DY holdings Inc. - [2024.04 - 2025.03] Tohoku UniversityGP-DS Research Assistant (Competitive research fellowship) - [2023.09] NS Solutions R&D Internship - [2023.10 - 2024.02] [AKATSUKI-SICA](https://mitouteki.jp/r4/supporters/outline/r4_b07/)([Certificate](https://www.openbadge-global.com/ns/portal/openbadge/public/assertions/detail/U3NWU05wcHViK2VHc3RSYTJZeFVhZz09))Social Impact Creators' Accelerator Program (supported by Ministry of Economy, Trade and Industry of Japan) - [2023.05 - 2024.01] [AI王](https://sites.google.com/view/project-aio/competition4?pli=1)([YouTube](https://youtu.be/5pT5t6e_bLo), [News: 東洋経済](https://toyokeizai.net/articles/-/732641?page=5), [News: Tech+](https://news.mynavi.jp/techplus/article/20240206-2877452/?&utm_medium=email&utm_campaign=20240213))Committee Member - [2021.03 - 2023.08] Infratop (DMM WebCamp)Programming Mentor, School Managemenent Member  ## Invited Talks         {% for talk in site.oubInvitedTalks reversed %}                       {% for speaker in talk.speakers %}           {{ speaker.name }}           {%- if forloop.last == false -%}             ,           {% endif %}         {%- endfor -%}         .         {{ talk.title }}.         {{ talk.event_name }},         {{ talk.month }}         {{ talk.year }}.         {% if talk.links %}           [           {%- for link in talk.links -%}             {{ link.name}}{% if forloop.last == false %}, {% endif %}           {%- endfor -%}           ]         {% endif %}            {% endfor %}         ","url": "/cv/"
  },{
    "title": "",
    "excerpt":"","url": "/index.html"
  },{
    "title": " - page 2",
    "excerpt":"","url": "/page/2/index.html"
  }]
