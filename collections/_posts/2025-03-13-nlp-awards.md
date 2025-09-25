---
layout: splash
title: "👑 2 papers received awards at NLP 2025"
category: News
excerpt: "若手奨励賞 + 日本経済新聞社 CDIO室賞"
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
---
言語処理学会 第31回年次大会 (NLP 2025) にて、以下の2件の論文がそれぞれ「若手奨励賞」「日本経済新聞社 CDIO室賞」を受賞しました。

<div>
  <ul>
    <li>
      {% assign publication = site.pubDomesticConferences | where: "title", "層の冗長性と層同士の独立性に基づく言語モデルの層交換の成否の特徴づけ" | first %}
      {% include pub-anlp-domestic-conf.html  publication=publication %}
    </li>
    <li>
      {% assign publication = site.pubDomesticConferences | where: "title", "LM は日本の時系列構造をどうエンコードするか" | first %}
      {% include pub-anlp-domestic-conf.html  publication=publication %}
    </li>
  </ul>
</div>

-> [Publications](/cv/)