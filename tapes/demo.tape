# Run from the repository root after `cargo build --release`: vhs tapes/demo.tape
# `notrieve init` downloads the preset model when it is not cached yet, and the download then ends
# up in the GIF: run any `notrieve init` once beforehand.
# The vault is a copy of tests/eval/vault, registered in a throwaway XDG_STATE_HOME.
Output tapes/demo.gif

Set Shell zsh
Set TypingSpeed 40ms

Set FontSize 28
Set Width 2200
Set Height 900

Hide
Type `demo=$(mktemp -d) && cp -R tests/eval/vault "$demo/notes"` Enter
Type `exec env XDG_STATE_HOME="$demo/state" PATH="$PWD/target/release:$PATH" zsh -f -c 'cd "$1/notes" && exec zsh -f' _ "$demo"` Enter
Type `PS1='$ '` Enter
Type "clear" Enter
Show

Sleep 500ms
Type "notrieve init" Sleep 1s
Enter Sleep 3s

Hide
Type "clear" Enter
Show

Type "notrieve search -k 2 'シカに囲まれた旅の思い出'" Sleep 1s
Enter Sleep 4s

Type "notrieve search -k 2 'サービスが止まったとき真っ先にやること'" Sleep 1s
Enter Sleep 5s
