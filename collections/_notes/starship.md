---
layout: splash
title: "Starshipの設定"
header:
  teaser: /assets/img/starship.png
  show_overlay_excerpt: false
  overlay_color: "#59876F"
show_date: true
date: 2026-11-24
excerpt: "Macで使っているStarshipとNerd Fontの設定。"
---

<style>
.page__content pre,
.page__content code,
.page__content code span {
  font-family: "MesloLGL Nerd Font", "MesloLGL Nerd Font Mono", "MesloLGL NF", "MesloLGL NFM", monospace;
}
</style>

時刻、Python・Conda環境、ディレクトリ、Gitのブランチ・変更状況を表示する。現在のMacの設定をそのまま載せておく。

1. **Starshipを入れる**
   1. ターミナルで実行する。
      ```sh
      brew install starship
      ```
1. **フォントを入れてiTerm2に設定する**
   1. アイコンや区切り文字を正しく表示し、文字化け（□など）を防ぐためにNerd Fontを入れる。
      ```sh
      brew install --cask font-meslo-lg-nerd-font
      ```
   2. iTerm2 → Settings →「Profiles」→「Text」→「Font」で「MesloLGL Nerd Font」の Regular、12pt を選ぶ。
1. **zshでStarshipを有効にする**
   1. `~/.zshrc` に追加する。`zsh-syntax-highlighting` を読み込む行より前に置く。
      ```zsh
      # starship
      eval "$(starship init zsh)"
      ```
1. **設定ファイルを作る**
   1. `mkdir -p ~/.config` を実行し、以下の内容を `~/.config/starship.toml` に保存する。既に設定がある場合はバックアップしてから置き換える。
   2. ターミナルで新しいタブを開くと反映される。

※ MesloLGL Nerd FontがPCに入っていれば、以下のコード内のアイコンも表示されます。未導入の場合は、□（豆腐）になることがあります。

```toml
#### icon
# nf-ple-left_half_circle_thick
# nf-pl-left_hard_divider
# nf-pl-left_hard_divider
# nf-pl-right_hard_divider
# nf-ple-right_half_circle_thick

format = """
[](fg:#7aa2f7)\
$os\
$time\
$conda\
$python\
[ ](fg:#7aa2f7 bg:#1a1b26)\
$direnv\
$directory\
$git_branch\
$git_status\
$git_metrics\
[](fg:#1a1b26)\
\n$character\
"""

[os]
format = "[$symbol]($style)"
style = 'fg:#1a1b26 bg:#7aa2f7'
disabled = false

[os.symbols]
Macos = "   " # nf-fa-apple
Ubuntu = "  " # nf-linux-ubuntu
Debian = "  " # nf-linux-debian

[directory]
truncation_length = 6
truncation_symbol = ' ' # nf-fa-folder_open
truncate_to_repo = false
home_symbol = ' ~' # nf-costum-home
style = 'fg:#7aa2f7 bg:#1a1b26'
read_only = ' 󰌾 ' # nf-md-lock
read_only_style = 'fg:#f7768e bg:#1a1b26'
format = '[$path]($style)[$read_only]($read_only_style)'

[git_branch]
symbol = '  ' # nf-fa-github_alt, nf-fa-code_fork
truncation_length = 4
truncation_symbol = ''
style = 'fg:#7aa2f7 bg:#1a1b26'
format = '[  $symbol$branch(:$remote_branch)]($style)' # nf-pl-left_soft_divider

[git_status]
style = 'fg:#e0af68 bg:#1a1b26'
conflicted = '='
ahead = '⇡${count}'
behind = '⇣${count}'
diverged = '⇕'
up_to_date = '✓'
untracked = '?'
stashed = '$'
modified = '!${count}'
renamed = '»'
deleted = '✘'
format = '([\[$all_status$ahead_behind\]]($style))'

[git_metrics]
added_style = 'fg:#9ece6a bg:#1a1b26'
deleted_style = 'fg:#9ece6a bg:#1a1b26'
format = '[+$added/-$deleted]($deleted_style)'
disabled = false

[fill]
symbol = '─'
style = 'blue'

[conda]
symbol = '  ' # nf-dev-python
style = 'fg:#9ece6a bg:#1a1b26'
format = '[ $symbol$environment ]($style)'
ignore_base = false

[python]
symbol = '  ' # nf-dev-python
format = '[ ${symbol}${pyenv_prefix}(${version})(\($virtualenv\)) ]($style)'
pyenv_version_name = false
style = 'fg:#1a1b26 bg:#7aa2f7'

[direnv]
format = '[$symbol$allowed]($style) '
style = "bold fg:#1a1b26 bg:#cba6f7"
disabled = false

[character]
success_symbol = '[❯](bold #9ece6a)'
error_symbol = '[❯](bold red)'

[cmd_duration]
min_time = 1
style = 'fg:#e0af68'
format = "[   $duration]($style)" # nf-pl-right_soft_divider, nf-mdi-clock

[lua]
symbol = "" # nf-seti-lua
format = '[  $symbol $version](blue)' # nf-pl-right_soft_divider

[rust]
symbol = "" # nf-dev-rust
format = '[  $symbol $version](red)' # nf-pl-right_soft_divider

[time]
disabled = false
style = 'fg:#73daca'
format = '[  $time ]($style)' # nf-pl-right_soft_divider, nf-fa-clock_o
time_format = '%T'
utc_time_offset = '+9'
```

時刻は `utc_time_offset = '+9'` で日本時間に固定している。

[Starship公式ガイド](https://starship.rs/guide/) · [MesloLG Nerd FontのHomebrewページ](https://formulae.brew.sh/cask/font-meslo-lg-nerd-font)


