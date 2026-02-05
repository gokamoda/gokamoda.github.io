---
layout: splash
title: "VSCodeで開くボタンをMacのFinderに追加する"
use_math: true
header:
  teaser: /assets/img/open-in-vscode.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-01-21
excerpt: "VSCodeで開くボタンをMacのFinderに追加する方法のメモ"
---

1. Automatorを開く
1. 画面上部のメニューバーで「File」->「New」を選択
1. 「Quick Action」を選択
1. 「Workflow receives current」のドロップダウンメニューで「files or folders」を選択
1. 「in」ドロップダウンメニューで「Finder.app」を選択
1. 左側のライブラリから「Run Shell Script」をダブルクリックし，画面右側に追加
1. 「Pass input:」のドロップダウンメニューで「as arguments」を選択
1. 以下のスクリプトを入力
    ```bash
    for f in "$@"
    do
        open -a "Visual Studio Code" "$f"
    done
    ```
1. 画面上部のメニューバーで「File」->「Save」を選択
1. 「Save As:」に「Open in VSCode」などと入力し，「Save」ボタンをクリック


※ Terminalからなら`code`コマンドを使うのが早い．