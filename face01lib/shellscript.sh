#!/usr/bin/env bash
set -Ceux -o pipefail
IFS=$'\n\t'

# set
# C: リダイレクトで既存のファイルを上書きしない
# e: exit status not equal 0 -> terminate script
# u: 初期化していない変数があるとエラー(特殊パラメーターである「@」と「*」は除く）
# x: 実行するコマンドを出力して何をしたらどうなったかがログに残る(トレース情報として、シェルが実行したコマンドとその引数を出力する。情報の先頭にはシェル変数PS4の値を使用)
# o: when error occurred on pipeline, terminate the script
# -o pipefail: パイプの途中で発生したエラーがexit codeとなる。デフォルトではパイプ最後のコマンドのexit code。
# IFS=$'\n\t': 引数の区切り文字は改行とタブのみに指定。（空白は区切り文字に含めない）

function my_command() {


    return 1
}

function my_error() {
    zenity --error --text="\


    "
    exit 1
}

my_command || my_error