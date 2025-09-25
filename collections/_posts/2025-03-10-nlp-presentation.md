---
layout: splash
title: "🎤 Presentations at NLP 2025"
category: News
excerpt: "6 presentations"
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
---
言語処理学会 第31回年次大会 (NLP 2025) にて、以下の6件の発表があります。

<div>
  <ul>
    {% assign publications = site.pubDomesticConferences | where: "proceedings_title", "言語処理学会 第31回年次大会" %}
    {% for publication in publications %}
    <li>
      {% include pub-anlp-domestic-conf.html  publication=publication hideawards=true %}
    </li>
    {% endfor %}
  </ul>
</div>

-> [Publications](/cv/)