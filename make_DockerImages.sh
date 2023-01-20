#!/usr/bin/env bash
set -Ceux -o pipefail
IFS=$'\n\t'


# Automation docker operation from local build to push to DockerHub.

# TODO: face01_gpuとface01_no_gpuをfor loopでまわすこと

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

# cd: DIST/
cd ~/bin/DIST


# ////////////////////////////////////////

# face01_gpu

# ////////////////////////////////////////

# docker build: CPU100%になるので他の作業との兼ね合いに注意すること
docker build -t face01_gpu:1.4.11 -f docker/Dockerfile_gpu . --network host
# get `container-id`
face01_gpu_container-id = docker ps -a | grep face01_gpu:1.4.11 | awk '{print $1}'
# commit
docker container commit "${face01_gpu_container-id}" tokaikaoninsho/face01_gpu:1.4.11
# get `image-id`
face01_gpu_image-id = docker images | grep -E "tokaikaoninsho/face01_gpu\s+1.4.11.*" | awk '{print $3}'
# add tag
docker tag "${face01_gpu_image-id}" face01_gpu
# login
docker login
# docker push
docker push tokaikaoninsho/face01_gpu:1.4.11


# ////////////////////////////////////////

# face01_no_gpu

# ////////////////////////////////////////

# docker build: CPU100%になるので他の作業との兼ね合いに注意すること
docker build -t face01_no_gpu:1.4.11 -f docker/Dockerfile_no_gpu . --network host
# get `container-id`
face01_no_gpu-container-id = docker ps -a | grep face01_no_gpu:1.4.11 | awk '{print $1}'
# commit
docker container commit "${face01_no_gpu_container-id}" tokaikaoninsho/face01_no_gpu:1.4.11
# get `image-id`
face01_no_gpu_image-id = docker images | grep -E "tokaikaoninsho/face01_no_gpu\s+1.4.11.*" | awk '{print $3}'
# add tag
docker tag "${face01_no_gpu_image-id}" face01_no_gpu
# login
docker login
# docker push
docker push tokaikaoninsho/face01_no_gpu:1.4.11


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

#   Docker
#       [docker push 手順](https://zenn.dev/katan/articles/1d5ff92fd809e7)
#       [grep, awkによる抽出](https://zenn.dev/sickleaf/articles/99884a12b0489cf21d45)
#   google style guide
#       [Shell Style Guide](https://github.com/google/styleguide/blob/gh-pages/shellguide.md)
#       [Googleの肩に乗ってShellコーディングしちゃおう](https://qiita.com/ma91n/items/5f72ca668f1c58176644)


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