---
layout: splash
title: "Macを最初に買ったらやること"
header:
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-11-10
excerpt: "新しいMacで最初にやる7つの設定。"
---

1. **ソフトウェアアップデート**
   1. 「システム設定」→「一般」→「ソフトウェアアップデート」で更新する。
1. **Homebrewをインストールする**
   1. [Homebrew公式サイト](https://brew.sh/)からコマンドをコピーしてターミナルで実行する。
   2. その間に以下設定をやってしまう。
1. **Touch ID**
   1. Settings →「Touch ID & Password」
1. **キーボード入力を最速に**
   1. Settings →「Keyboard」→「Keyboard Repeat」→「Fastest」
   2. Settings →「Keyboard」→「Delay Until Repeat」→「Short」
1. **Dockの位置とサイズを変える**
   1. Settings →「Desktop & Dock」→「Size」→「Small」
   2. Settings →「Desktop & Dock」→「Dock Position on Screen」→「Left」
   3. Settings →「Desktop & Dock」→「Automatically hide and show the Dock」をオンにする。
1. **タップでクリックをオンにする**
   1. Settings →「Trackpad」→「Point & Click」→「Tap to click」をオンにする。
1. **Homebrewでアプリを入れる**
   1. ターミナルで次のコマンドを実行する。
      ```sh
      brew install --cask google-japanese-ime hammerspoon raycast iterm2 google-chrome stats
      ```
    - Google IME: こっちのほうが好みだった
    - Hammerspoon: USキーボードで日本語入力をするために使用。Karabiner-Elementsでも可だが以前Logi系との相性が悪かった。
    - Raycast: Spotlightの代替。クリップボード履歴、ウィンドウ切り替え、ファイル検索などが便利。
    - iTerm2: App HotKey が便利 (ショートカットキーでアプリの表示/非表示ができる)
    - Google Chrome: ブラウザ
    - Stats: CPUやメモリの使用率をメニューバーに表示するアプリ。
1. ログイン項目に追加するアプリを設定する。
   1. Settings →「General」→「Login Items & Extensions」→「+」で追加する。
   2. 追加するアプリ: Hammerspoon, Raycast, Stats
1. **デスクトップを5つ作ってショートカットを設定する**
   1. `ctrl + ↑` で Mission Control を開き、画面上部の「+」でデスクトップが合計5つになるまで追加する。
   2. Settings →「Keyboard」→「Keyboard Shortcuts」→「Mission Control」で「Switch to Desktop 1」〜「Switch to Desktop 5」をオンにし、それぞれ `ctrl + 1`〜`ctrl + 5` を割り当てる。
1. **起動音をオフにする**
   1. Settings →「Sound」→「Play sound on startup」をオフにする。
1. たぶんここで一旦再起動するとよい。
1. **google-japanese-imeの設定**
   1. settings →「Keyboard」→「Input Sources」→「Edit」→「+」→「Japanese」→「Hiragana (Google)」
   2. settings →「Keyboard」→「Input Sources」→「Edit」→「-」で「Japanese - Kana」
1. **Hammerspoonの設定**
   1. メニューバーのHammerspoonアイコンをクリックして「Open Config」を選択する。 
   2. [Hammerspoon設定](./cmd-eikana.md)を参照してconfig.luaを作成する。
   3. メニューバーのHammerspoonアイコンをクリックして「Reload Config」を選択する。 
1. **Raycastの設定**
   1. スポットライトショートカット無効: settings → 「Keyboard」 → 「Keyboard Shortcuts」 → 「Spotlight」 → 「Show Spotlight search」をオフにする。
   2. RayCastを起動すると、ショートカット設定画面が表示されるので、cmd + space を設定する。
   3. クリップボード履歴: Raycast Settings →「Shortcuts」で「Clipboard History」の Hotkey を `cmd + shift + v` にする。
   4. Window Management: Raycast Settings →「Shortcuts」→「Window Management」で、各コマンドの Hotkey を設定する。
      1. Left Half（左半分） → `cmd + ctrl + ←`
      2. Right Half（右半分） → `cmd + ctrl + →`
      3. Top Half（上半分） → `cmd + ctrl + ↑`
      4. Bottom Half（下半分） → `cmd + ctrl + ↓`
      5. Maximize（画面いっぱいに広げる） → `cmd + ctrl + Enter`
      6. Top Right Quarter（右上） → `cmd + ctrl + shift + →`
      7. Top Left Quarter（左上） → `cmd + ctrl + shift + ↑`
      8. Bottom Left Quarter（左下） → `cmd + ctrl + shift + ←`
      9. Bottom Right Quarter（右下） → `cmd + ctrl + shift + ↓`
1. **iTerm2の設定**
   1. iTerm2 → Settings →「Keys」→「Hotkey」→「Show/hide iTerm2 with a system-wide hotkey」をオンにする。
1. **zshの入力候補と構文ハイライトを設定する**
   1. ターミナルでプラグインをインストールする。
      ```sh
      brew install zsh-autosuggestions zsh-syntax-highlighting
      ```
   2. `~/.zshrc` の末尾に追加する。入力履歴から候補が表示され、コマンドに色が付く。
      ```zsh
      # zsh
      source "$(brew --prefix)/share/zsh-autosuggestions/zsh-autosuggestions.zsh"
      source "$(brew --prefix)/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh"
      ```
   3. ターミナルで新しいタブを開くと反映される。`zsh-syntax-highlighting` はほかのプラグインより最後に読み込む。
1. **Starshipの設定**
   1. [Starshipの設定]({{ "/notes/starship/" | relative_url }})を参照して、Starshipを導入し、プロンプトの見た目を整える。
2. VSCode
3. 
4. tmux
