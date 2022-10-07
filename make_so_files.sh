#!/usr/bin/env bash
set -Ceux -o pipefail
IFS=$'\n\t'


# ./face01lib/*.pyをsoファイルへ変換する

# TODO: GitHub Actionsに書き換える

#######################################
# __SUMMARY__
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   None
#######################################
function my_command() {

# ////////////////////////////////////////

# 処理説明補足

# ////////////////////////////////////////


cp -f ~/bin/FACE01/face01lib/*.py \
    ~/bin/FACE01/face01lib/python_files/


mv ~/bin/FACE01/face01lib/*.py \
    ~/bin/FACE01/face01lib/pyx/


rm ~/bin/FACE01/face01lib/pyx/*.cpython-38-x86_64-linux-gnu.so
rm ~/bin/FACE01/face01lib/pyx/*.c
rm ~/bin/FACE01/face01lib/pyx/*.html


cd ~/bin/FACE01/face01lib/pyx
rename 's/.py/.pyx/' ./*.py


cp ~/bin/FACE01/face01lib/python_files/compile.py \
    ~/bin/FACE01/face01lib/pyx/


# venv activation
source ~/bin/FACE01/bin/activate

python ~/bin/FACE01/face01lib/pyx/compile.py build_ext --inplace


mv ~/bin/FACE01/face01lib/pyx/build/lib.linux-x86_64-cpython-38/pyx/*  \
    ~/bin/FACE01/face01lib/


rm ~/bin/FACE01/face01lib/__init__*.so
rm ~/bin/FACE01/face01lib/compile*.so
touch ~/bin/FACE01/face01lib/__init__.py


cp -f  ~/bin/FACE01/face01lib/*.cpython-38-x86_64-linux-gnu.so \
    ~/bin/DIST/face01lib/


zenity --info --text="exampleをテストしてください"

python ~/bin/FACE01/example/display_GUI_window.py


# 避難させていたpyファイルをface01lib/へ戻す
rm ~/bin/FACE01/face01lib/*.cpython-38-x86_64-linux-gnu.so
mv ~/bin/FACE01/face01lib/python_files/*.py \
    ~/bin/FACE01/face01lib/


pactl set-sink-volume @DEFAULT_SINK@ 30%
play -v 0.4 -q ~/done.wav

return 0
}


function my_error() {
    zenity --error --text="\
    失敗しました。
    "
    exit 1
}

my_command || my_error


# ////////////////////////////////////////
# REFERENCE:




# ////////////////////////////////////////
# MEMORANDUM:

# google style guide
#       [Shell Style Guide](https://github.com/google/styleguide/blob/gh-pages/shellguide.md)
#       [Googleの肩に乗ってShellコーディングしちゃおう](https://qiita.com/ma91n/items/5f72ca668f1c58176644)

# set
#   C: リダイレクトで既存のファイルを上書きしない
#   e: exit status not equal 0 -> terminate script
#   u: 初期化していない変数があるとエラー(特殊パラメーターである「@」と「*」は除く）
#   x: 実行するコマンドを出力して何をしたらどうなったかがログに残る(トレース情報として、シェルが実行したコマンドとその引数を出力する。情報の先頭にはシェル変数PS4の値を使用)
#   o: when error occurred on pipeline, terminate the script
#   -o pipefail: パイプの途中で発生したエラーがexit codeとなる。デフォルトではパイプ最後のコマンドのexit code。
#   IFS=$'\n\t': 引数の区切り文字は改行とタブのみに指定。（空白は区切り文字に含めない）