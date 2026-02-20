---
layout: splash
title: "Publications & Activities"
permalink: /cv/
author_profile: false
excerpt: ""
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
---

## Education
- <strong>Doctoral Student / 博士後期課程</strong> [2025.04 - ]<br> [Graduate Institute for Advanced Studies, SOKENDAI.](https://www.soken.ac.jp/en/) / [総合研究大学院大学](https://www.soken.ac.jp/)<br>Supervisor: Assoc. Prof. Sho Yokoi<br>
- <strong>Master of Information Science / 修士（情報科学）</strong> [2023.04 - 2025.03]<br>[Graduate School of Information Sciences, Tohoku University.](https://www.is.tohoku.ac.jp/en/) / [東北大学大学院情報科学研究科](https://www.is.tohoku.ac.jp/)<br>Supervisor: Prof. Jun Suzuki & Assoc. Prof. Keisuke Sakaguchi<br><i class="fa-solid fa-award" style="color: #5d9679;"></i> [Dean Award / 研究科長賞](https://www.is.tohoku.ac.jp/jp/activity/award/detail---id-1593.html) (4/126)
- <strong>Bachelor of Engineering / 学士（工学）</strong> [2020.04 - 2023.03]<br>[School of Engineering, Tohoku University.](https://www.eng.tohoku.ac.jp/english/) / [東北大学工学部](https://www.eng.tohoku.ac.jp/)<br>Supervisor: Prof. Kentaro Inui & Assoc. Prof. Keisuke Sakaguchi<br><i class="fa-solid fa-award" style="color: #5d9679;"></i>Early Graduation / 早期卒業 (1/252)

## International Conferences

<div>
  <ul>
    {% for publication in site.pubInternationalConferences reversed %}
    <li>
        {% include pub-international-conf.html  %}
      </li>
    {% endfor %}

  </ul>

</div>




## Domestic Conferences

<div>
  <ul>
    {% for publication in site.pubDomesticConferences reversed %}
    <li>
        {% include pub-anlp-domestic-conf.html  %}
      </li>
    {% endfor %}

  </ul>

</div>

{% if site.pubPreprint.size > 0 %}
## Preprints
<div>
  <ul>
    {% for publication in site.pubPreprint reversed %}
    <li>
        {% include pub-international-conf.html  %}
      </li>
    {% endfor %}

  </ul>
</div>
{% endif %}

{% if site.experiences.size > 0 %}
## Experiences
<div>
  <ul>
    {% for experience in site.experiences reversed %}
    <li>
        {% include experiences.html  %}
      </li>
    {% endfor %}

  </ul>
</div>
{% endif %}

<!-- ## Invited Talks -->
<div>
  <ol>
    {% for talk in site.oubInvitedTalks reversed %}
    <li>
        <!-- https://www.anlp.jp/guide/guideline.html -->
        {% for speaker in talk.speakers %}
          {{ speaker.name }}
          {%- if forloop.last == false -%}
            ,
          {% endif %}
        {%- endfor -%}
        .
        {{ talk.title }}.
        {{ talk.event_name }},
        {{ talk.month }}
        {{ talk.year }}.
        {% if talk.links %}
          [
          {%- for link in talk.links -%}
            <a href="{{ link.url }}" target="_blank">{{ link.name}}</a>{% if forloop.last == false %}, {% endif %}
          {%- endfor -%}
          ]
        {% endif %}
      </li>
    {% endfor %}

  </ol>

</div>


