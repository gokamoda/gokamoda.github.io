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
- Doctoral Student, April 2025 - <br>Graduate Institute for Advanced Studies, SOKENDAI.<br>Supervisor: Assoc. Prof. Sho Yokoi<br>
- Master of Information Science, April 2023 - March 2025.<br>Graduate School of Information Sciences, Tohoku University.<br>Supervisor: Prof. Jun Suzuki & Assoc. Prof. Keisuke Sakaguchi<br><i class="fa-solid fa-award" style="color: #5d9679;"></i> Dean Award (4/126)
- Bachelor of Engineering, April 2020 - March 2023.<br>School of Engineering, Tohoku University.<br>Supervisor: Prof. Kentaro Inui & Assoc. Prof. Keisuke Sakaguchi<br><i class="fa-solid fa-award" style="color: #5d9679;"></i>Early Graduation (1/252)

## International Conferences

<div>
  <ul>
    {% for publication in site.pubInternationalConferences reversed %}
    <li>
        {% include pub-apa-international-conf.html  %}
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
        {% include pub-apa-international-conf.html  %}
      </li>
    {% endfor %}

  </ul>
</div>
{% endif %}


## Experiences
- [2025.04 -] SOKENDAI<br>Special Researcher Program (Supported by JST BOOST)
- [2025.04 -] NINJAL<br>Part-time Researcher
- [2023.10 -] Joint Research with Hakuhodo DY holdings Inc.
- [2024.04 - 2025.03] Tohoku University<br>GP-DS Research Assistant (Competitive research fellowship)
- [2023.09] NS Solutions <br>R&D Internship
- [2023.10 - 2024.02] [AKATSUKI-SICA](https://mitouteki.jp/r4/supporters/outline/r4_b07/)([Certificate](https://www.openbadge-global.com/ns/portal/openbadge/public/assertions/detail/U3NWU05wcHViK2VHc3RSYTJZeFVhZz09))<br>Social Impact Creators' Accelerator Program (supported by Ministry of Economy, Trade and Industry of Japan)
- [2023.05 - 2024.01] [AI王](https://sites.google.com/view/project-aio/competition4?pli=1)([YouTube](https://youtu.be/5pT5t6e_bLo), [News: 東洋経済](https://toyokeizai.net/articles/-/732641?page=5), [News: Tech+](https://news.mynavi.jp/techplus/article/20240206-2877452/?&utm_medium=email&utm_campaign=20240213))<br>Committee Member
- [2021.03 - 2023.08] Infratop (DMM WebCamp)<br>Programming Mentor, School Managemenent Member

## Invited Talks
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


