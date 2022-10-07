#!/usr/bin/env bash
set -Ceux -o pipefail
IFS=$'\n\t'


# Synchronize the FACE01 folder and the DIST folder using rsync command.

# TODO: 

#######################################
# __SUMMARY__
# Globals:
#   BACKUP_DIR
#   ORACLE_SID
# Arguments:
#   None
# Returns:
#   None
#######################################
function my_command() {
cd ~/bin/FACE01;


# フォルダの同期 ※face01libディレクトリ以下は同期させないこと！
rsync -r -t --progress -u -l -H -s --exclude-from='exclude-file.txt' ~/bin/FACE01/ ~/bin/DIST/
rsync -r -t --progress -u -l -H -s --exclude-from='exclude-file.txt' ~/bin/DIST/ ~/bin/FACE01/


# git cacheの更新
git add ./FACE01/*
git add ../DIST/*


    return 0
}

function my_error() {
    zenity --error --text="\
    Synchronization between FACE01 folder and DIST folder was failed.
    "
    exit 1
}

my_command || my_error


# ////////////////////////////////////////
# REFERENCE:
#    [除外リストファイルを用意して[–exclude-from]オプションを使用する](https://minory.org/rsync-exclude.html#exclude-from-%E9%99%A4%E5%A4%96%E3%83%AA%E3%82%B9%E3%83%88%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E5%90%8D)

# ////////////////////////////////////////
# MEMORANDUM:

# set
#   C: リダイレクトで既存のファイルを上書きしない
#   e: exit status not equal 0 -> terminate script
#   u: 初期化していない変数があるとエラー(特殊パラメーターである「@」と「*」は除く）
#   x: 実行するコマンドを出力して何をしたらどうなったかがログに残る(トレース情報として、シェルが実行したコマンドとその引数を出力する。情報の先頭にはシェル変数PS4の値を使用)
#   o: when error occurred on pipeline, terminate the script
#   -o pipefail: パイプの途中で発生したエラーがexit codeとなる。デフォルトではパイプ最後のコマンドのexit code。
#   IFS=$'\n\t': 引数の区切り文字は改行とタブのみに指定。（空白は区切り文字に含めない）
