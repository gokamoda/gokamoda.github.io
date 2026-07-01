---
layout: splash
title: "Publications & Activities"
permalink: /cv/
author_profile: false
excerpt: ""
classes:
  - cv-page
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
---

<style>
  .cv-print-header {
    display: none;
  }

  body.cv-page .link-row .link-btn + .link-btn {
    margin-left: 0.25em;
  }

  .cv-toc {
    border-bottom: 1px solid #d8dee4;
    border-top: 1px solid #d8dee4;
    margin: 0.25rem 0 0.5rem;
    padding: 0.45rem 0;
  }

  .cv-toc__list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1rem;
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .cv-actions {
    margin: 0.25rem 0 0.25rem;
    text-align: left;
    margin-bottom: 0rem !important;
  }

  .cv-toc__list li {
    margin: 0;
  }

  .cv-toc__list a {
    font-size: 0.86rem;
    text-decoration: none;
  }

  .cv-toc__list a:hover {
    text-decoration: underline;
  }

  @page {
    margin: 14mm 13mm;
    size: A4;
  }

  @media print {
    html {
      font-size: 11px;
    }

    body.cv-page {
      background: #fff !important;
      color: #000 !important;
      font-size: 1rem;
      line-height: 1.32;
    }

    body.cv-page .page__hero,
    body.cv-page .page__hero--overlay,
    body.cv-page .page__hero-image,
    body.cv-page .breadcrumbs,
    body.cv-page .masthead,
    body.cv-page .page__footer,
    body.cv-page .no-print {
      display: none !important;
    }

    body.cv-page #main,
    body.cv-page .splash,
    body.cv-page .page__content {
      margin: 0;
      max-width: none;
      padding: 0;
      width: 100%;
    }

    body.cv-page .cv-print-header {
      border-bottom: 2px solid #59876f;
      display: block;
      margin: 0 0 0.75rem;
      padding: 0 0 0.45rem;
    }

    body.cv-page .cv-print-header__title {
      font-size: 1.75rem;
      line-height: 1.1;
      margin: 0 0 0.25rem;
    }

    body.cv-page .cv-print-header__meta {
      color: #444;
      font-size: 0.9rem;
      line-height: 1.2;
    }

    body.cv-page h1,
    body.cv-page h2,
    body.cv-page h3,
    body.cv-page h4,
    body.cv-page h5,
    body.cv-page h6 {
      color: #000;
      line-height: 1.2;
      break-after: auto;
      break-inside: auto;
      page-break-after: auto;
      page-break-inside: auto;
    }

    body.cv-page h1 {
      font-size: 1.75rem;
      margin: 0 0 0.75rem;
    }

    body.cv-page h2 {
      border-bottom: 1px solid #d8dee4;
      font-size: 1rem;
      margin: 0.8rem 0 0.45rem;
      padding-bottom: 0.15rem;
    }

    body.cv-page h3,
    body.cv-page h4,
    body.cv-page h5,
    body.cv-page h6 {
      font-size: 0.95rem;
      margin: 0.65rem 0 0.35rem;
    }

    body.cv-page ul,
    body.cv-page ol {
      break-inside: auto;
      margin-bottom: 0.65rem;
      page-break-inside: auto;
      padding-left: 1.15rem;
    }

    body.cv-page li {
      break-inside: auto;
      margin-bottom: 0.38rem;
      page-break-inside: auto;
    }

    body.cv-page a,
    body.cv-page .link-row,
    body.cv-page .link-btn {
      break-inside: auto;
      page-break-inside: auto;
    }

    body.cv-page.cv-export-pdf .link-row .link-btn {
      border-color: darkgray;
      border-radius: 5px;
      border-style: solid;
      border-width: 1px;
      color: #000;
      display: inline-block;
      margin-bottom: 0.25em;
      padding: 5px 10px;
      text-decoration: none;
    }

    body.cv-page.cv-export-pdf .link-row .link-btn + .link-btn {
      margin-left: 0.25em;
    }

    body.cv-page:not(.cv-export-pdf) .link-row {
      align-items: stretch;
      display: block;
      margin-top: 0.5rem;
      margin-left: 1rem;
    }

    body.cv-page:not(.cv-export-pdf) .link-row .link-btn {
      background: transparent !important;
      border: 0 !important;
      border-radius: 0 !important;
      color: #000;
      display: block;
      font-family: inherit;
      font-size: inherit;
      font-weight: inherit;
      line-height: 1.05;
      margin: 0;
      padding: 0 !important;
      text-align: left;
      text-decoration: none;
      white-space: normal;
      margin-bottom: 0.3rem;
    }

    body.cv-page .link-row .link-btn img {
      display: inline-block !important;
      margin: 0 0.15em 0 0 !important;
      vertical-align: middle;
    }

    body.cv-page .link-row .link-btn i {
      display: inline-block !important;
      flex: 0 0 auto;
    }

    body.cv-page:not(.cv-export-pdf) .link-row .link-btn[href]::after {
      content: " (" attr(href) ")" !important;
      font-size: 0.86em;
      line-height: 1.15;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    body.cv-page.cv-export-pdf a[href]::after,
    body.cv-page.cv-export-pdf a[href^='http://']::after,
    body.cv-page.cv-export-pdf a[href^='https://']::after,
    body.cv-page.cv-export-pdf a[href^='ftp://']::after,
    body.cv-page.cv-export-pdf .link-row .link-btn[href]::after {
      content: none !important;
    }
  }
</style>

<header class="cv-print-header" aria-hidden="true">
  <h1 class="cv-print-header__title">Go Kamoda</h1>
  <div class="cv-print-header__meta">
    {{ site.author.email }}{% if site.url %} · {{ site.url }}{% endif %}
  </div>
</header>

<p class="cv-actions no-print">
  <button class="btn btn--primary" type="button" onclick="printCv('pdf')">Export as PDF</button>
  <button class="btn btn--primary" type="button" onclick="printCv('paper')">Print</button>
</p>

<nav class="cv-toc no-print" aria-label="CV sections">
  <ul class="cv-toc__list">
    <li><a href="#education">Education</a></li>
    <li><a href="#international-conferences">International Conferences</a></li>
    <li><a href="#domestic-conferences">Domestic Conferences</a></li>
    {% if site.pubPreprint.size > 0 %}
    <li><a href="#preprints">Preprints</a></li>
    {% endif %}
    {% if site.experiences.size > 0 %}
    <li><a href="#experiences">Experiences</a></li>
    {% endif %}
  </ul>
</nav>

<script>
  function printCv(mode) {
    document.body.classList.remove('cv-export-pdf', 'cv-paper-print');
    document.body.classList.add(mode === 'pdf' ? 'cv-export-pdf' : 'cv-paper-print');
    window.print();
  }

  window.addEventListener('afterprint', function () {
    document.body.classList.remove('cv-export-pdf', 'cv-paper-print');
  });
</script>

{% include cv-content.html %}
