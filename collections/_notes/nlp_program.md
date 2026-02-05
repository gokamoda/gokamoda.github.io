---
layout: splash
title: "言語処理学会年次大会のプログラムスクショ"
use_math: true
header:
  teaser: /assets/img/nlp_program.png
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

![スクショ例]({{ "/assets/img/nlp_program.png" | relative_url }})



## DevTools コンソール用スクリプト

- 使用方法
  1. 年次大会のプログラムページを開く
  2. ブラウザのDevToolsを開く (画面右クリック → 検証/Inspect など)
  3. コンソールタブに移動
  4. 下のコードを丸ごとコピー＆ペーストしてEnterキーを押す (TARGET_NAMEは適宜変更すること．) 



```js
(() => {
  const TARGET_NAME = "鴨田 豪";

  /* ------------------------------
   * 0. ロゴ周りの微調整 (お好みで．)
   * ------------------------------ */

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

  /* ------------------------------
   * ① 索引テーブル → 対象ID集合
   * ------------------------------ */
  const targetIds = new Set(
    [...document.querySelectorAll("tr")]
      .filter(tr => tr.innerText.includes(TARGET_NAME))
      .flatMap(tr =>
        [...tr.querySelectorAll('a[href^="#"]')]
          .map(a => a.getAttribute("href").slice(1))
      )
  );
  if (!targetIds.size) return;

  /* ------------------------------
   * ② 本文側を上から順に走査（時系列保証）
   * ------------------------------ */
  const entries = [];
  document.querySelectorAll("td.pid span[id]").forEach(pidSpan => {
    const id = pidSpan.id;
    if (!targetIds.has(id)) return;

    const titleTr = pidSpan.closest("tr");
    const authorTr = titleTr?.nextElementSibling;
    const table = titleTr?.closest("table");

    let header = table?.previousElementSibling;
    while (header && !header.classList?.contains("session_header")) {
      header = header.previousElementSibling;
    }

    entries.push({
      id,
      title: titleTr?.querySelector(".title")?.innerText ?? "",
      authors: authorTr?.innerText ?? "",
      sessionNode: header
    });
  });
  if (!entries.length) return;

  /* ------------------------------
   * ③ セッション単位でグルーピング（出現順維持）
   * ------------------------------ */
  const sessions = new Map();
  const sessionOrder = [];
  for (const e of entries) {
    const key = e.sessionNode;
    if (!sessions.has(key)) {
      sessions.set(key, []);
      sessionOrder.push(key);
    }
    sessions.get(key).push(e);
  }

  /* ------------------------------
   * ④ 表示ブロック生成（装飾なし）
   * ------------------------------ */
  const box = document.createElement("div");
  box.style.cssText = `margin: 16px 0 24px 0;`;

  for (const sessionNode of sessionOrder) {
    const papers = sessions.get(sessionNode);

    const sessionDiv = document.createElement("div");
    sessionDiv.style.marginBottom = "9px";
    sessionDiv.style.paddingTop = "9px";
    sessionDiv.style.borderTop = "1px solid #ccc";

    sessionDiv.appendChild(sessionNode.cloneNode(true));

    const ul = document.createElement("ul");
    ul.style.marginTop = "6px";

    papers.forEach(p => {
      const li = document.createElement("li");
      li.innerHTML = `
        <b><a href="#${p.id}">${p.id}</a> ${p.title}</b><br>
        <small>${p.authors}</small>
      `;
      ul.appendChild(li);
    });

    sessionDiv.appendChild(ul);
    box.appendChild(sessionDiv);
  }

  /* ------------------------------
   * ⑤ フッター（小さく・一番下）
   * ------------------------------ */
  const footer = document.createElement("div");
  footer.textContent = `${TARGET_NAME} の発表一覧`;
  footer.style.cssText = `
    font-size: 12px;
    color: #666;
    text-align: right;
    margin-top: 8px;
  `;
  box.appendChild(footer);

  /* ------------------------------
   * ⑥ ロゴ直後に挿入
   * ------------------------------ */
  const logo = document.getElementById("logo");
  if (logo) {
    logo.insertAdjacentElement("afterend", box);
  } else {
    document.body.prepend(box);
  }
})();

```


## Bookmarklet の場合

- Chromeで動作確認済み (2026/02/04)
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

  /* 索引テーブル → 対象ID集合 */
  const targetIds = new Set(
    [...document.querySelectorAll("tr")]
      .filter(tr => tr.innerText.includes(TARGET_NAME))
      .flatMap(tr =>
        [...tr.querySelectorAll('a[href^="#"]')]
          .map(a => a.getAttribute("href").slice(1))
      )
  );
  if (!targetIds.size) {
    alert("該当する発表が見つかりませんでした");
    return;
  }

  /* 本文側を上から順に走査（時系列保証） */
  const entries = [];
  document.querySelectorAll("td.pid span[id]").forEach(pidSpan => {
    const id = pidSpan.id;
    if (!targetIds.has(id)) return;

    const titleTr = pidSpan.closest("tr");
    const authorTr = titleTr?.nextElementSibling;
    const table = titleTr?.closest("table");

    let header = table?.previousElementSibling;
    while (header && !header.classList?.contains("session_header")) {
      header = header.previousElementSibling;
    }

    entries.push({
      id,
      title: titleTr?.querySelector(".title")?.innerText ?? "",
      authors: authorTr?.innerText ?? "",
      sessionNode: header
    });
  });

  if (!entries.length) {
    alert("本文側で発表が見つかりませんでした");
    return;
  }

  /* セッション単位でグルーピング（出現順維持） */
  const sessions = new Map();
  const sessionOrder = [];
  for (const e of entries) {
    const key = e.sessionNode;
    if (!sessions.has(key)) {
      sessions.set(key, []);
      sessionOrder.push(key);
    }
    sessions.get(key).push(e);
  }

  /* 表示ブロック生成 */
  const box = document.createElement("div");
  box.style.margin = "16px 10px 24px 10px";

  for (const sessionNode of sessionOrder) {
    const papers = sessions.get(sessionNode);

    const sessionDiv = document.createElement("div");
    sessionDiv.style.marginBottom = "9px";
    sessionDiv.style.paddingTop = "9px";
    sessionDiv.style.borderTop = "1px solid #ccc";

    sessionDiv.appendChild(sessionNode.cloneNode(true));

    const ul = document.createElement("ul");
    ul.style.marginTop = "6px";

    papers.forEach(p => {
      const li = document.createElement("li");
      li.innerHTML =
        "<b><a href='#" + p.id + "'>" + p.id + "</a> " + p.title + "</b><br>" +
        "<small>" + p.authors + "</small>";
      ul.appendChild(li);
    });

    sessionDiv.appendChild(ul);
    box.appendChild(sessionDiv);
  }

  /* フッター */
  const footer = document.createElement("div");
  footer.textContent = TARGET_NAME + " の発表一覧";
  footer.style.fontSize = "12px";
  footer.style.color = "#666";
  footer.style.textAlign = "right";
  footer.style.marginTop = "8px";
  box.appendChild(footer);

  /* ロゴ直後に挿入（既存があれば削除して差し替え） */
  const old = document.getElementById("__personal_program_box");
  if (old) old.remove();
  box.id = "__personal_program_box";

  const logo = document.getElementById("logo");
  if (logo) {
    logo.insertAdjacentElement("afterend", box);
  } else {
    document.body.prepend(box);
  }
})();
```