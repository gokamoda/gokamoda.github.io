---
layout: splash
title: "言語処理学会年次大会のプログラムスクショ"
use_math: true
header:
  teaser: /assets/img/nlp2026_program_oral.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-02-04
excerpt: "自分の発表部分を抜き出すためのスクリプトメモ"
---

## これはなに
- 言語処理学会年次大会のプログラムページから，自分の発表部分だけを抜き出して表示するためのスクリプトメモ
- 発表者名を指定して実行すると，該当発表だけを抽出して，ページ上部に一覧表示する
- DevToolsコンソール用スクリプトとBookmarkletの2種類を用意
- スクショのために，ロゴと大会名の部分を少し小さくする微調整も含む
  - ※以下画像は、スペーシングを後から微調整している 

{: align="center"}
![スクショ例]({{ "/assets/img/nlp2026_program_oral.png" | relative_url }})


## DevTools コンソール用スクリプト

- 使用方法
  1. 年次大会のプログラムページを開く
  2. ブラウザのDevToolsを開く (画面右クリック → 検証/Inspect など)
  3. コンソールタブに移動
  4. 下のコードを丸ごとコピー＆ペーストしてEnterキーを押す (TARGET_NAMEは適宜変更すること．) 


```js
(() => {
  const TARGET_NAME = "鴨田 豪";

  // ロゴ画像を少し小さく
  const logoImg = document.querySelector("#logo img");
  if (logoImg) {
    logoImg.style.width = "180px";
    logoImg.style.height = "auto";
  }

  // h1 を h3 より少し大きい程度に
  const logoH1 = document.querySelector("#logo h1");
  if (logoH1) {
    logoH1.style.fontSize = "1.25em";
    logoH1.style.fontWeight = "600";
    logoH1.style.marginBottom = "6px";
  }

  function parseSessionDate(sessionNode) {
    if (!sessionNode) return new Date(9999, 0, 1);
    const text = sessionNode.innerText;
    const m = text.match(/(\d+)\/(\d+).*?(\d+):(\d+)/);
    if (!m) return new Date(9999, 0, 1);
    return new Date(
      2026,
      parseInt(m[1], 10) - 1,
      parseInt(m[2], 10),
      parseInt(m[3], 10),
      parseInt(m[4], 10)
    );
  }
  function formatAuthors(text) {
    if (!text) return "";

    // 全角スペース → 半角
    let s = text.replace(/\u3000/g, " ");

    // 一旦すべてのスペース削除
    s = s.replace(/ /g, "");

    // カンマの後にスペースを入れ直す
    s = s.replace(/,/g, ", ");

    return s;
  }

  function formatSessionHeader(sessionNode) {
  const raw = sessionNode.innerText.replace(/\s+/g, " ").trim();

  const timeMatch = raw.match(/\d+\/\d+.*?\d+:\d+-\d+:\d+/);
  const chairMatch = raw.match(/座長:.+$/);
  const sessionIdTitleMatch = raw.match(/^[A-Z]\d+:.+?(?=\d+\/\d+)/);

  let locationMatch = null;
  if (timeMatch) {
    const afterTime = raw.slice(
      raw.indexOf(timeMatch[0]) + timeMatch[0].length
    );
    locationMatch = afterTime.match(/^[^座長]+/);
  }

  const wrapper = document.createElement("div");

  // 1行目：時間 + 会場
  const line1 = document.createElement("div");
  line1.style.fontWeight = "600";
  line1.textContent =
    (timeMatch?.[0] ?? "") +
    "  " +
    (locationMatch?.[0]?.trim() ?? "");
  wrapper.appendChild(line1);

  // 2行目：セッション名 + 座長
  const line2 = document.createElement("div");
  line2.style.marginTop = "2px";

  let sessionLine = sessionIdTitleMatch?.[0] ?? "";
  if (chairMatch) {
    sessionLine += " | " + chairMatch[0];
  }

  line2.textContent = sessionLine;
  wrapper.appendChild(line2);

  return wrapper;
}

  /* ------------------------------
   * 索引解析
   * ------------------------------ */
  const posterToOral = new Map();
  const oralToPoster = new Map();
  const posterIds = new Set();
  const oralSessionIds = new Set();

  [...document.querySelectorAll("tr")]
    .filter(tr => tr.innerText.includes(TARGET_NAME))
    .forEach(tr => {
      const links = [...tr.querySelectorAll('a[href^="#"]')];
      let lastPosterId = null;

      for (const link of links) {
        const id = link.getAttribute("href").slice(1);

        if (/^[A-Z]\d+-\d+$/.test(id)) {
          lastPosterId = id;
          posterIds.add(id);
        } else if (/^A\d+$/.test(id) && lastPosterId) {
          posterToOral.set(lastPosterId, id);
          oralToPoster.set(id, lastPosterId);
          oralSessionIds.add(id);
        }
      }
    });

  const entries = [];

  /* ------------------------------
   * ポスター抽出
   * ------------------------------ */
  document.querySelectorAll("td.pid span[id]").forEach(span => {
    const id = span.id;
    if (!posterIds.has(id)) return;

    const tr = span.closest("tr");
    const authorTr = tr?.nextElementSibling;
    const table = tr?.closest("table");

    let header = table?.previousElementSibling;
    while (header && !header.classList?.contains("session_header")) {
      header = header.previousElementSibling;
    }

    entries.push({
      kind: "poster",
      id,
      oralSession: posterToOral.get(id),
      title: (tr?.querySelector(".title")?.innerText ?? "")
        .replace(/\bORAL\b/gi, "")
        .trim(),
      authors: authorTr?.innerText ?? "",
      pdf: authorTr?.querySelector('a[href$=".pdf"]')?.href ?? "",
      sessionNode: header
    });
  });

  /* ------------------------------
   * 口頭抽出
   * ------------------------------ */
  oralSessionIds.forEach(sessionId => {
    const headerSpan =
      document.querySelector(`.session_title#${sessionId}`);
    if (!headerSpan) return;

    const sessionDiv = headerSpan.closest(".session_header");
    const table = sessionDiv?.nextElementSibling;
    if (!table) return;

    table.querySelectorAll("td.pid span[id]").forEach(span => {
      const tr = span.closest("tr");
      const authorTr = tr?.nextElementSibling;
      if (!authorTr?.innerText.includes(TARGET_NAME)) return;

      let title =
        tr?.querySelector(".title")?.innerText ??
        tr?.innerText ??
        "";

      title = title.replace(/\bORAL\b/gi, "").trim();

      entries.push({
        kind: "oral",
        id: span.id,
        posterId: oralToPoster.get(sessionId),
        oralSession: sessionId,
        title,
        authors: authorTr?.innerText ?? "",
        pdf: authorTr?.querySelector('a[href$=".pdf"]')?.href ?? "",
        sessionNode: sessionDiv
      });
    });
  });

  if (!entries.length) return;

  const sessions = new Map();
  entries.forEach(e => {
    if (!sessions.has(e.sessionNode)) {
      sessions.set(e.sessionNode, []);
    }
    sessions.get(e.sessionNode).push(e);
  });

  const sortedSessions = [...sessions.keys()].sort(
    (a, b) => parseSessionDate(a) - parseSessionDate(b)
  );

  const box = document.createElement("div");
  box.style.margin = "16px 0 24px 0";

  sortedSessions.forEach(sessionNode => {
    const sessionDiv = document.createElement("div");
    sessionDiv.style.cssText =
      "margin-bottom:8px;padding-top:8px;border-top:1px solid #ccc;";

    sessionDiv.appendChild(formatSessionHeader(sessionNode));

    sessions.get(sessionNode).forEach(p => {
      const pdfPart = p.pdf
        ? ` <a href="${p.pdf}" target="_blank" style="margin-left:6px;">📄</a>`
        : "";

      let headerText = "";

      if (p.kind === "poster") {
        headerText = `
          📰 <a href="#${p.id}">${p.id}</a>
          ${p.oralSession ? ` (<a href="#${p.oralSession}">Oral</a>)` : ""}
        `;
      } else {
        headerText = `
          🎤 <a href="#${p.oralSession}">Oral</a>
          ${p.posterId ? ` (<a href="#${p.posterId}">${p.posterId}</a>)` : ""}
        `;
      }

      const item = document.createElement("div");
      item.style.marginTop = "8px";
      item.innerHTML = `
        <div>
          <b>${headerText} ${p.title}${pdfPart}</b>
        </div>
        <div style="font-size:0.7em;color:#555;">
          ${formatAuthors(p.authors)}
        </div>
      `;
      sessionDiv.appendChild(item);
    });

    box.appendChild(sessionDiv);
  });

  const logo = document.getElementById("logo");
  logo ? logo.after(box) : document.body.prepend(box);
})();
```


## Bookmarklet の場合

- Chromeで動作確認済み (2026/03/02)
- 導入方法
  1. 任意のページでBookmarkを作成
  2. BookmarkのURL部分に下のコードを丸ごとコピー＆ペースト
  3. 年次大会のプログラムページを開き，Bookmarkをクリック


```js

javascript:(() => {

  const TARGET_NAME = prompt("発表者名を入力してください", "姓 名");
  if (!TARGET_NAME) return;

  /* ロゴ微調整 */
  const logoImg = document.querySelector("#logo img");
  if (logoImg) {
    logoImg.style.width = "180px";
    logoImg.style.height = "auto";
  }

  const logoH1 = document.querySelector("#logo h1");
  if (logoH1) {
    logoH1.style.fontSize = "1.25em";
    logoH1.style.fontWeight = "600";
    logoH1.style.marginBottom = "6px";
  }


  /* ==============================
   * 日時パース
   * ============================== */
  function parseSessionDate(sessionNode) {
    if (!sessionNode) return new Date(9999, 0, 1);

    const text = sessionNode.innerText;
    const m = text.match(/(\d+)\/(\d+).*?(\d+):(\d+)/);
    if (!m) return new Date(9999, 0, 1);

    return new Date(
      2026,
      parseInt(m[1], 10) - 1,
      parseInt(m[2], 10),
      parseInt(m[3], 10),
      parseInt(m[4], 10)
    );
  }

  /* ==============================
   * 著者整形
   * ============================== */
  function formatAuthors(text) {
    if (!text) return "";

    let s = text.replace(/\u3000/g, " ");
    s = s.replace(/ /g, "");
    s = s.replace(/,/g, ", ");

    return s;
  }

  /* ==============================
   * セッション整形
   * ============================== */
  function formatSessionHeader(sessionNode) {
    const raw = sessionNode.innerText.replace(/\s+/g, " ").trim();

    const timeMatch = raw.match(/\d+\/\d+.*?\d+:\d+-\d+:\d+/);
    const chairMatch = raw.match(/座長:.+$/);
    const sessionMatch = raw.match(/^[A-Z]\d+:.+?(?=\d+\/\d+)/);

    let locationMatch = null;
    if (timeMatch) {
      const afterTime = raw.slice(
        raw.indexOf(timeMatch[0]) + timeMatch[0].length
      );
      locationMatch = afterTime.match(/^[^座長]+/);
    }

    const wrapper = document.createElement("div");

    const line1 = document.createElement("div");
    line1.style.fontWeight = "600";
    line1.textContent =
      (timeMatch?.[0] ?? "") +
      "  " +
      (locationMatch?.[0]?.trim() ?? "");
    wrapper.appendChild(line1);

    const line2 = document.createElement("div");
    let sessionLine = sessionMatch?.[0] ?? "";
    if (chairMatch) sessionLine += " | " + chairMatch[0];
    line2.textContent = sessionLine;
    wrapper.appendChild(line2);

    return wrapper;
  }

  /* ==============================
   * 索引解析
   * ============================== */
  const posterToOral = new Map();
  const oralToPoster = new Map();
  const posterIds = new Set();
  const oralSessionIds = new Set();

  [...document.querySelectorAll("tr")]
    .filter(tr => tr.innerText.includes(TARGET_NAME))
    .forEach(tr => {

      const links = [...tr.querySelectorAll('a[href^="#"]')];
      let lastPosterId = null;

      for (const link of links) {
        const id = link.getAttribute("href").slice(1);

        if (/^[A-Z]\d+-\d+$/.test(id)) {
          lastPosterId = id;
          posterIds.add(id);
        }
        else if (/^A\d+$/.test(id) && lastPosterId) {
          posterToOral.set(lastPosterId, id);
          oralToPoster.set(id, lastPosterId);
          oralSessionIds.add(id);
        }
      }

    });

  const entries = [];

  /* ==============================
   * ポスター抽出
   * ============================== */
  document.querySelectorAll("td.pid span[id]").forEach(span => {

    const id = span.id;
    if (!posterIds.has(id)) return;

    const tr = span.closest("tr");
    const authorTr = tr?.nextElementSibling;
    const table = tr?.closest("table");

    let header = table?.previousElementSibling;
    while (header && !header.classList?.contains("session_header")) {
      header = header.previousElementSibling;
    }

    entries.push({
      kind: "poster",
      id,
      oralSession: posterToOral.get(id),
      title: (tr?.querySelector(".title")?.innerText ?? "")
        .replace(/\bORAL\b/gi, "")
        .trim(),
      authors: authorTr?.innerText ?? "",
      pdf: authorTr?.querySelector('a[href$=".pdf"]')?.href ?? "",
      sessionNode: header
    });

  });

  /* ==============================
   * 口頭抽出
   * ============================== */
  oralSessionIds.forEach(sessionId => {

    const headerSpan =
      document.querySelector(`.session_title#${sessionId}`);
    if (!headerSpan) return;

    const sessionDiv = headerSpan.closest(".session_header");
    const table = sessionDiv?.nextElementSibling;
    if (!table) return;

    table.querySelectorAll("td.pid span[id]").forEach(span => {

      const tr = span.closest("tr");
      const authorTr = tr?.nextElementSibling;

      if (!authorTr?.innerText.includes(TARGET_NAME)) return;

      let title =
        tr?.querySelector(".title")?.innerText ??
        tr?.innerText ??
        "";

      title = title.replace(/\bORAL\b/gi, "").trim();

      entries.push({
        kind: "oral",
        id: span.id,
        posterId: oralToPoster.get(sessionId),
        oralSession: sessionId,
        title,
        authors: authorTr?.innerText ?? "",
        pdf: authorTr?.querySelector('a[href$=".pdf"]')?.href ?? "",
        sessionNode: sessionDiv
      });

    });

  });

  if (!entries.length) {
    alert("該当発表が見つかりませんでした。");
    return;
  }

  /* ==============================
   * セッションごとに整理
   * ============================== */
  const sessions = new Map();

  entries.forEach(e => {
    if (!sessions.has(e.sessionNode)) {
      sessions.set(e.sessionNode, []);
    }
    sessions.get(e.sessionNode).push(e);
  });

  const sortedSessions =
    [...sessions.keys()].sort(
      (a, b) => parseSessionDate(a) - parseSessionDate(b)
    );

  /* ==============================
   * 表示生成
   * ============================== */
  const box = document.createElement("div");
  box.style.margin = "16px 0 24px 0";

  sortedSessions.forEach(sessionNode => {

    const sessionDiv = document.createElement("div");
    sessionDiv.style.cssText =
      "margin-bottom:8px;padding-top:8px;border-top:1px solid #ccc;";

    sessionDiv.appendChild(formatSessionHeader(sessionNode));

    sessions.get(sessionNode).forEach(p => {

      const pdfPart =
        p.pdf
          ? ` <a href="${p.pdf}" target="_blank" style="margin-left:6px;">📄</a>`
          : "";

      let headerText = "";

      if (p.kind === "poster") {
        headerText =
          `📰 <a href="#${p.id}">${p.id}</a>` +
          (p.oralSession
            ? ` (<a href="#${p.oralSession}">Oral</a>)`
            : "");
      }
      else {
        headerText =
          `🎤 <a href="#${p.oralSession}">Oral</a>` +
          (p.posterId
            ? ` (<a href="#${p.posterId}">${p.posterId}</a>)`
            : "");
      }

      const item = document.createElement("div");
      item.style.marginTop = "8px";

      item.innerHTML = `
        <div>
          <b>${headerText} ${p.title}${pdfPart}</b>
        </div>
        <div style="font-size:0.7em;color:#555;">
          ${formatAuthors(p.authors)}
        </div>
      `;

      sessionDiv.appendChild(item);

    });

    box.appendChild(sessionDiv);

  });

  const logo = document.getElementById("logo");
  logo ? logo.after(box) : document.body.prepend(box);

})();
```