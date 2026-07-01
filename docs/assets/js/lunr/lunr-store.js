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
        "title": "ACL Rolling Review (ARR) Reviewer",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202505-arr-reviewer",
        "teaser": null
      },{
        "title": "LLM-JP ファインチューニングコンペ (Team dentaku)",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202603-ft-llm",
        "teaser": null
      },{
        "title": "NLP若手の会 (YANS) Committee Member",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202603-yans",
        "teaser": null
      },{
        "title": "HHKB配列分割キーボード",
        "excerpt":"※7sProを2023年9月に購入し、ケーブルなどなど揃い、キースイッチ情報等消えそうなので2025/12/01に備忘録的にまとめた 背景 HHKBを1台持っているが、ラボと家の両方で使いたい 分割キーボード、気になる 腕を広げたまま使えて負担が減りそう 真ん中にiPadとか置けて便利そう iPadで論文表示して、PCでメモ取るとか HHKBを2台買うほど、HHKBが最適解だと思えていない 未開拓のキーボードは多数あるので とはいえ、HHKBのいいところはなるべく維持したい 打鍵感 (何によるものかは本当にはわかっていない) 配列・コンパクトさ 個人的に US配列 かつ Enter 上に Backspace があるのが好き ANSI配列でBackspaceのところにキーが2つあるのも好き 結果 + 選択理由 自作キーボードキット 7sPro プログラマーやプロフェッショナルの方に愛用者の多いハッピーな配列を踏襲している 似たものでChoco60もあるが、違いはあまりわかってない。一つ確かなのは、7sProはスペース部分の分割が多い。 自作キーボードなら大体そうだろうが、Cherry MX互換スイッチが使える (選択肢が多い) はんだ付けは記憶が正しければ10箇所くらい キースイッチ Kailh Box Mute Brown Switch ※ おそらく廃盤で正式名称があやふや (2023/9/16は72個入りで7,356円だった) HHKBに近い HHKB Type-S: 静電容量無接点方式、キーストローク3.8mm、押下圧45g Kailh Box Mute Brown Switch:...","categories": [],
        "tags": [],
        "url": "/notes/split_hhkb/",
        "teaser": "/assets/img/7spro.jpg"
      },{
        "title": "RMSNorm and LayerNorm",
        "excerpt":"Preparation Let $\\mu(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}$ be a function that returns element-wise mean of a row-vector $\\bm{x} \\in \\mathbb{R}^{d}$: \\[\\begin{align} \\mu(\\bm{x}) &amp;= \\frac{1}{d}\\sum_{i=1}^d \\bm{x}_i\\\\ &amp;=\\frac{1}{d}\\bm{x}\\cdot \\bm{1}\\\\ &amp;=\\frac{1}{d}\\bm{x}\\bm{1}^\\top \\end{align}\\] Let $c(\\bm{x}): \\mathbb{R}^{d}\\rightarrow \\mathbb{R}^{d}$ be centering function, that subtracts the element-wise mean from each element of $\\bm{x}$: \\[\\begin{aligned} c(\\bm{x})&amp;=\\bm{x} - \\mu(\\bm{x})\\bm{1}\\\\ &amp;= \\bm{x}...","categories": [],
        "tags": [],
        "url": "/notes/layernorm/",
        "teaser": "/assets/img/layernorm_rmsnorm.png"
      },{
        "title": "LogitLens without bias",
        "excerpt":"LogitLens[1] applies $\\text{LMHead}$ to the internal representations $(\\bm{h} \\in \\mathbb{R}^{1\\times d})$ of a transformer model. \\[\\begin{equation} \\text{LMHead}(\\bm{h}) = \\text{LN}_\\text{f}(\\bm{h})\\bm{E}^O \\label{eq:lm_head} \\end{equation}\\] Here, $\\bm{E}^O \\in \\mathbb{R}^{d\\times |\\mathcal{V}|}$ is the unembedding matrix and $\\text{LN}_\\text{f}$ is the final layer normalization of a transformer model. In this page, we assume LayerNorm (not RMSNorm) is...","categories": [],
        "tags": [],
        "url": "/notes/logitlens-wob/",
        "teaser": "/assets/img/logit_lens_nobias_gpt2.png"
      },{
        "title": "Folding weights in transformers",
        "excerpt":"This reformulation of LayerNorm and Self-Attention is used in our paper: Go Kamoda, Benjamin Heinzerling, Tatsuro Inaba, Keito Kudo, Keisuke Sakaguchi, &amp; Kentaro Inui (2025). Weight-based Analysis of Detokenization in Language Models: Understanding the First Stage of Inference Without Inference. In Findings of the Association for Computational Linguistics: NAACL 2025....","categories": [],
        "tags": [],
        "url": "/notes/fold-weights/",
        "teaser": "/assets/img/folding-weights.png"
      },{
        "title": "GPT-OSS Attnention Sink",
        "excerpt":"tl;dr: The bias term for preventing attention sink in GPT-OSS may have other effects than just preventing attention sink. (made public on 2026-01-22) Notation \\[\\begin{alignat}{4} &amp;\\bm{X} &amp;:= &amp; \\begin{bmatrix} \\bm{x}_1\\\\ \\vdots\\\\ \\bm{x}_n \\end{bmatrix} &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{n \\times d}\\\\ &amp;\\bm{W}^O &amp;:= &amp; \\begin{bmatrix} \\bm{W}^O_1\\\\ \\vdots\\\\ \\bm{W}^O_H \\end{bmatrix} &amp;\\hspace{1em}\\in &amp;\\mathbb{R}^{d \\times d}\\\\ &amp;\\bm{W}^Q...","categories": [],
        "tags": [],
        "url": "/notes/oss/",
        "teaser": "/assets/img/gptoss-bias.png"
      },{
        "title": "PCA",
        "excerpt":"$n$ 個の $d$ 次元のデータ $\\bm{X}$ を、$d’$ 次元に圧縮するための線形変換行列 $\\bm{W}$ を求める。 \\[\\bm{X} =\\begin{bmatrix} \\bm{x}_1 \\\\ \\bm{x}_2 \\\\ \\vdots \\\\ \\bm{x}_n \\end{bmatrix} = \\begin{bmatrix} x_{1,1} &amp; x_{1,2} &amp; \\cdots &amp; x_{1,d} \\\\ x_{2,1} &amp; x_{2,2} &amp; \\cdots &amp; x_{2,d} \\\\ \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots \\\\ x_{n,1} &amp; x_{n,2} &amp; \\cdots...","categories": [],
        "tags": [],
        "url": "/notes/pca/",
        "teaser": "/assets/img/pca.png"
      },{
        "title": "RoPE Implementation",
        "excerpt":"From paper Su et al. (2020) Let $\\bm{q}_m$ be the query vector at position $m$ before applying RoPE, and $\\mathring{\\bm{q}}_m$ be the query vector after applying RoPE. \\[\\begin{align} \\begin{bmatrix} \\mathring{q}_m^{(1)} \\\\ \\mathring{q}_m^{(2)} \\\\ \\mathring{q}_m^{(3)} \\\\ \\mathring{q}_m^{(4)} \\\\ \\vdots \\\\ \\mathring{q}_m^{(d-1)}\\\\ \\mathring{q}_m^{(d)} \\end{bmatrix} &amp;= \\begin{bmatrix} \\cos m\\theta_1 &amp; -\\sin m\\theta_1 &amp;...","categories": [],
        "tags": [],
        "url": "/notes/rope-implementation/",
        "teaser": "/assets/img/rotate-half.png"
      },{
        "title": "torch.nn.functional.kl_div",
        "excerpt":"メモの目的 以下の問いへの回答を書いておく $D_{\\text{KL}}(P||Q)$ の、PとQはどっちが予測分布でどっちが正解分布か torch.nn.functional.kl_div の引数には何を渡せばよいのか 一般的な式 \\[D_{KL}(P || Q) = \\sum_{i} P(i) \\log\\frac{P(i)}{Q(i)}\\] a measure of how much an approximating probability distribution Q is different from a true probability distribution P [Wikipedia (2025-12-15)] (他の参考書を参照したほうが良いのだが、まぁ。) つまり、$P$ を真の分布 (正解分布)、$Q$ を予測分布とする。 torch.nn.functional.kl_div の引数 結論 input は予測分布の対数を渡す target は正解分布を渡す 公式ドキュメント より (2025-12-15, v2.9.1):...","categories": [],
        "tags": [],
        "url": "/notes/kl_div/",
        "teaser": "/assets/img/kl_div.png"
      },{
        "title": "VSCodeで開くボタンをMacのFinderに追加する",
        "excerpt":"   Automatorを開く   画面上部のメニューバーで「File」-&gt;「New」を選択   「Quick Action」を選択   「Workflow receives current」のドロップダウンメニューで「files or folders」を選択   「in」ドロップダウンメニューで「Finder.app」を選択   左側のライブラリから「Run Shell Script」をダブルクリックし，画面右側に追加   「Pass input:」のドロップダウンメニューで「as arguments」を選択   以下のスクリプトを入力      for f in \"$@\"  do      open -a \"Visual Studio Code\" \"$f\"  done           画面上部のメニューバーで「File」-&gt;「Save」を選択   「Save As:」に「Open in VSCode」などと入力し，「Save」ボタンをクリック   ※ Terminalからならcodeコマンドを使うのが早い．  ","categories": [],
        "tags": [],
        "url": "/notes/mac-vscode/",
        "teaser": "/assets/img/open-in-vscode.png"
      },{
        "title": "言語処理学会年次大会のプログラムスクショ",
        "excerpt":"これはなに 言語処理学会年次大会のプログラムページから，自分の発表部分だけを抜き出して表示するためのスクリプトメモ 発表者名を指定して実行すると，該当発表だけを抽出して，ページ上部に一覧表示する DevToolsコンソール用スクリプトとBookmarkletの2種類を用意 スクショのために，ロゴと大会名の部分を少し小さくする微調整も含む ※以下画像は、スペーシングを後から微調整している DevTools コンソール用スクリプト 使用方法 年次大会のプログラムページを開く ブラウザのDevToolsを開く (画面右クリック → 検証/Inspect など) コンソールタブに移動 下のコードを丸ごとコピー＆ペーストしてEnterキーを押す (TARGET_NAMEは適宜変更すること．) (() =&gt; { const TARGET_NAME = \"鴨田 豪\"; // ロゴ画像を少し小さく const logoImg = document.querySelector(\"#logo img\"); if (logoImg) { logoImg.style.width = \"180px\"; logoImg.style.height = \"auto\"; } // h1 を h3 より少し大きい程度に const logoH1...","categories": [],
        "tags": [],
        "url": "/notes/nlp_program/",
        "teaser": "/assets/img/nlp2026_program_oral.png"
      },{
        "title": "cmd英かな on Hammerspoon",
        "excerpt":"参考: Tashiro Yutaka: Hammerspoon で英数・かなの切り替えを行う -- BEGIN CODE FOR CMD-EIKANA local map = hs.keycodes.map local keyDown = hs.eventtap.event.types.keyDown local flagsChanged = hs.eventtap.event.types.flagsChanged local keyStroke = hs.eventtap.keyStroke local isCmdAsModifier = false local function switchInputSourceEvent(event) local eventType = event:getType() local keyCode = event:getKeyCode() local flags = event:getFlags() local isCmd = flags['cmd'] if...","categories": [],
        "tags": [],
        "url": "/notes/cmd-eikana/",
        "teaser": "/assets/img/cmd-eikana.png"
      },{
        "title": "One-hot target と確率分布",
        "excerpt":"TL;DR 言語モデルの次トークン予測では各位置の教師は one-hot だが、データ全体での負の対数尤度 (NLL) 最小化は、各文脈における経験的な次トークン分布へモデル分布を近づけることとして理解できる。 Cross Entropy Loss 言語モデルの学習は、通常、コーパス中の各位置における次トークンの負の対数尤度を最小化することとして定式化される。 データセットを文脈と次トークンの組 \\[\\mathcal{D}=\\{(x_i,y_i)\\}_{i=1}^{M}\\] と書くと、負の対数尤度は \\[\\mathcal{L}(\\theta) = -\\sum_{i=1}^{M}\\log p_\\theta(y_i\\mid x_i)\\label{eq:nll}\\] である。 (平均 loss として定義する場合は、これを $M$ で割ればよいが、以下の最適化解には影響しない。) ここで、特定の文脈 $x$ の直後にトークン $y$ が出現する回数を $N(x,y)$ とし、文脈 $x$ の総出現回数を \\[N(x)=\\sum_{y'}N(x,y')\\] とする。 全体の損失 $\\mathcal{L}(\\theta)$ を、文脈 $x$ ごとにまとめると、 \\[\\begin{align} \\mathcal{L}(\\theta) &amp;= -\\sum_x\\sum_y N(x,y)\\log p_\\theta(y\\mid x)\\\\ &amp;= -\\sum_x N(x)\\sum_y...","categories": [],
        "tags": [],
        "url": "/notes/one-hot-vs-probability/",
        "teaser": "/assets/img/one-hot-vs-prob.png"
      },{
        "title": "✅ Paper accepted to the COLING 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to COLING 2025. Go Kamoda, Akari Asai, Ana Brassard, &amp; Keisuke Sakaguchi (2025). Quantifying the Influence of Evaluation Aspects on Long-Form Response Assessment. In Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025). ACL Anthology Google Scholar...","categories": ["News"],
        "tags": [],
        "url": "/news/coling2025-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the ICLR 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to ICLR 2025. Hiroki Deguchi, Go Kamoda, Yusuke Matsushita, Chihiro Taguchi, Masaki Waga, Kohei Suenaga, &amp; Sho Yokoi (2025). SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches. In The Thirteenth International Conference on Learning Representations (ICLR...","categories": ["News"],
        "tags": [],
        "url": "/news/iclr2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentations at NLP 2025",
        "excerpt":"言語処理学会 第31回年次大会 (NLP 2025) にて、以下の6件の発表があります。 出口祥之, 鴨田豪, 松下祐介, 田口智大, 末永幸平, 和賀正樹, 横井祥 (2024). SoftMatcha: 大規模コーパス検索のための柔らかくも高速なパターンマッチャー. 言語処理学会 第31回年次大会, pp. 3310-3315. 予稿 鴨田豪, Benjamin Heinzerling, 稲葉達郎, 工藤慧音, 坂口慶祐, 乾健太郎 (2025). 言語モデルのパラメータから探るDetokenizationメカニズム. 言語処理学会 第31回年次大会, pp. 634-639. 予稿 小林春斗, 原知正, 鴨田豪, 横井祥 (2025). 層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ. 言語処理学会 第31回年次大会, pp. 1751-1756. 予稿 工藤慧音, 鴨田豪, 塩野大輝, 鈴木潤 (2025)....","categories": ["News"],
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
        "excerpt":"We are pleased to announce that our paper has been accepted to the Findings of EMNLP 2025. Tatsuro Inaba, Go Kamoda, Kentaro Inui, Masaru Isonuma, Yusuke Miyao, Yohei Oseki, Yu Takagi, &amp; Benjamin Heinzerling (2025). How a Bilingual LM Becomes Bilingual: Tracing Internal Representations with Sparse Autoencoders. In Findings of...","categories": ["News"],
        "tags": [],
        "url": "/news/emnlp2025findings-accept/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the 8th BlackboxNLP Workshop",
        "excerpt":"We are pleased to announce that our paper has been accepted to the 8th BlackboxNLP Workshop. Ryosuke Takahashi, Go Kamoda, Benjamin Heinzerling, Keisuke Sakaguchi, &amp; Kentaro Inui (2025). Understanding the Side Effects of Rank-One Knowledge Editing. In BlackboxNLP 2025: The 8th Workshop on Analyzing and Interpreting Neural Networks for NLP....","categories": ["News"],
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
        "title": "✅ Paper accepted to the IJCNLP-AACL 2025",
        "excerpt":"We are pleased to announce that our paper has been accepted to the 8th BlackboxNLP Workshop. Mutsumi Sasaki, Go Kamoda, Ryosuke Takahashi, Kosuke Sato, Benjamin Heinzerling, Keisuke Sakaguchi, &amp; Kentaro Inui (2025). Can Language Models Handle a Non-Gregorian Calendar? The Case of the Japanese wareki. In Proceedings of the 14th...","categories": ["News"],
        "tags": [],
        "url": "/news/ijcnlpaacl2025-accept/",
        "teaser": null
      },{
        "title": "🎤 Presentation at NLP 2026",
        "excerpt":"NLP 2026 にて以下の発表があります。 [Update 2026/03/02]: 全て口頭発表に選出されました！ 米田優峻, 鴨田豪, 松下祐介, 末永幸平, 秋葉拓哉, 和賀正樹, 横井祥 (2026). SoftMatcha 2: 1兆語規模コーパスの超高速かつ柔らかい検索. 言語処理学会 第32回年次大会. 言語処理学会第32回年次大会 優秀賞 (13/789件) Project Page NLP 2026 プログラム 木谷頼斗, 大橋諭貴, 佐藤宏亮, 鴨田豪, 高橋良允, 山本悠士, 塩野大輝, 坂口慶祐, 小林悟郎 (2026). Attention Sink および Massive Activation の発生機序の分解. 言語処理学会 第32回年次大会. 言語処理学会第32回年次大会 若手奨励賞 (21/517件) NLP 2026...","categories": ["News"],
        "tags": [],
        "url": "/news/nlp2026-presentation/",
        "teaser": null
      },{
        "title": "👑 3 papers received awards at NLP 2026",
        "excerpt":"2026年3月9日から13日にかけて宇都宮（栃木）で開催された 言語処理学会第32回年次大会 (NLP 2026) にて、以下の研究で3件の受賞がありました。 3件の発表のうち、以下の賞を受賞しました。 優秀賞 米田優峻, 鴨田豪, 松下祐介, 末永幸平, 秋葉拓哉, 和賀正樹, 横井祥 (2026). SoftMatcha 2: 1兆語規模コーパスの超高速かつ柔らかい検索. 言語処理学会 第32回年次大会. 言語処理学会第32回年次大会 優秀賞 (13/789件) Project Page NLP 2026 プログラム 若手奨励賞 賞は第一著者に与えられる． 木谷頼斗, 大橋諭貴, 佐藤宏亮, 鴨田豪, 高橋良允, 山本悠士, 塩野大輝, 坂口慶祐, 小林悟郎 (2026). Attention Sink および Massive Activation の発生機序の分解. 言語処理学会 第32回年次大会. 言語処理学会第32回年次大会 若手奨励賞...","categories": ["News"],
        "tags": [],
        "url": "/news/nlp2026-awards/",
        "teaser": null
      },{
        "title": "🏆 Won 1st place in open-source division and 2nd place overall in the math task of the FT-LLM 2026 competition.",
        "excerpt":"言語処理学会第32回年次大会 (NLP 2026) にて開催された LLM-JP ファインチューニングコンペ (FT-LLM 2026) の数学タスクにおいて、所属する Team dentaku が以下の成績を収めました。      オープン部門 1位   総合 2位  ","categories": ["News"],
        "tags": [],
        "url": "/news/ft-llm/",
        "teaser": null
      },{
        "title": "✅ Paper accepted to the ICML 2026",
        "excerpt":"We are pleased to announce that our paper has been accepted to ICML 2026. Masataka Yoneda, Yusuke Matsushita, Go Kamoda, Kohei Suenaga, Takuya Akiba, Masaki Waga, &amp; Sho Yokoi (2026). SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora. In Forty-third International Conference on Machine Learning. Project Page...","categories": ["News"],
        "tags": [],
        "url": "/news/icml2026-accept/",
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
        "title": "Attention Sink および Massive Activation の発生機序の分解",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202603-nlp-kiya",
        "teaser": null
      },{
        "title": "注意機構における Attention Sink のバイアス項的解釈",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202603-nlp-ohashi",
        "teaser": null
      },{
        "title": "SoftMatcha 2: 1兆語規模コーパスの超高速かつ柔らかい検索",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202603-nlp-yoneda",
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
        "title": "Can Language Models Handle a Non-Gregorian Calendar? The Case of the Japanese wareki",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202512-aacl-sasaki",
        "teaser": null
      },{
        "title": "SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202607-icml-yoneda",
        "teaser": null
      },{
        "title": "Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "/202606-arxiv-sasaki",
        "teaser": null
      },{
    "title": "Notes",
    "excerpt":"","url": "https://gokamoda.github.io/notes/"
  },{
    "title": "Publications & Activities",
    "excerpt":"     Go Kamoda         {{ site.author.email }}{% if site.url %} · {{ site.url }}{% endif %}         Export as PDF   Print            Education     International Conferences     Domestic Conferences     {% if site.pubPreprint.size > 0 %}     Preprints     {% endif %}     {% if site.experiences.size > 0 %}     Experiences     {% endif %}        {% include cv-content.html %} ","url": "https://gokamoda.github.io/cv/"
  },{
    "title": "",
    "excerpt":"","url": "https://gokamoda.github.io/index.html"
  },{
    "title": " - page 2",
    "excerpt":"","url": "https://gokamoda.github.io/page/2/index.html"
  },{
    "title": " - page 3",
    "excerpt":"","url": "https://gokamoda.github.io/page/3/index.html"
  }]
