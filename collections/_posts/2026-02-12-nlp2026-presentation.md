---
layout: splash
title: "🎤 Presentation at NLP 2026"
category: News
excerpt: "NLP 2026 にて3件の発表があります。"
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
---

{: align="center"}
<img src='{{ "/assets/img/nlp2026_program.png" | relative_url }}' alt="NLP 2026 プログラム" style="max-width: 1000px; height: auto; padding: 0 auto; width: 100%;"/>





NLP 2026 にて以下の発表があります。
<div>
  <ul>
    <li>
    {% assign publication = site.pubDomesticConferences
      | where: "slug", "202603-nlp-yoneda"
      | first %}
      {% include pub-anlp-domestic-conf.html  publication=publication %}
    </li>
    <li>
      {% assign publication = site.pubDomesticConferences | where: "slug", "202603-nlp-kiya" | first %}
      {% include pub-anlp-domestic-conf.html  publication=publication %}
    </li>
    <li>
      {% assign publication = site.pubDomesticConferences
      | where: "slug", "202603-nlp-ohashi"
      | first %}
      {% include pub-anlp-domestic-conf.html  publication=publication %}
    </li>
  </ul>
</div>

-> [Publications](/cv/)


※ ちなみに、発表部分だけを抜き出してスクショを撮るためのスクリプトメモは [こちらのノート]( /notes/nlp_program/ ) にあります。
