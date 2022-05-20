import os
import shutil

import numpy as np

import face_recognition

# FACE01GRAPHICS127への対応 =========================================
# priset_face_imagesに新しい顔を登録した時にnpKnown.npzに反映されないバグをフィックス
#
#
# ===================================================================

# FACE01GRAPHICS124への対応 =========================================
# npKnown.txtからnpKnown.npzへの変更
# hogからcnnへの自動変換
#
# ===================================================================


def load_priset_image(
    kaoninshoDir,
    priset_face_imagesDir,
    upsampling=0,
    jitters=100,
    mode='hog',
    model='small'
):
    # ローカル変数の宣言と初期化
    known_face_names_list = []
    known_face_encodings_list = []
    new_files = []
    cnt = 1  # cnt：何番目のファイルかを表す変数

    os.chdir(kaoninshoDir)

    print("Loading npKnown.npz")
    print("...")

    ###################### npKnown.npzの有る無しで処理を分岐させる ######################
    # ============= npKnown.npzファイルが存在する場合の処理 ===============
    if os.path.exists("npKnown.npz"):

        # npKnown.npzの読み込みを行い、今までの全てのデータを格納する
        npKnown = np.load('npKnown.npz', allow_pickle=True)
        A, B = npKnown.files
        known_face_names_ndarray = npKnown[A]
        known_face_encodings_ndarray = npKnown[B]

        # ############ 各配列の整形（ndarray型からリスト型へ変換する） ############
        known_face_names_list = known_face_names_ndarray.tolist()

        list = []
        for i in known_face_encodings_ndarray:
            list.append(i)
            # for x in i:
            #     list.append(x)
        known_face_encodings_list = list
        # #########################################################################

        # priset_face_imagesフォルダ内の全てのファイル名を読み込む
        os.chdir(priset_face_imagesDir)
        # まずpriset_face_imagesDirのファイル名を全て得る
        for priset_face_image in os.listdir(priset_face_imagesDir):
            # <DEBUG>
            # if 'テスト' in priset_face_image:
            #     if not priset_face_image in known_face_names:
            #         print(priset_face_image)
            #     exit()
            # 関係ないファイルやフォルダは処理からとばす
            if priset_face_image == 'desktop.ini':
                continue
            if os.path.isdir(priset_face_image):
                continue
            if 'debug' in priset_face_image:
                continue
            # all_priset_face_images.append(priset_face_image)

            # =============== file名がnpKnownのキーに存在していない場合の処理 ===============
            if not priset_face_image in known_face_names_list:
                # priset_face_imageはknown_face_names配列にないから、new_fileに名前を変える
                new_file = priset_face_image

                new_file_face_image = face_recognition.load_image_file(
                    new_file)
                new_file_face_locations = face_recognition.face_locations(
                    new_file_face_image, upsampling, mode)

                # 顔検出できなかった場合hogからcnnへチェンジして再度顔検出する
                if len(new_file_face_locations) == 0:
                    if mode == 'hog':
                        print('顔が検出できませんでした。一時的にcnnモードへ切り替えます')
                        new_file_face_locations = face_recognition.face_locations(
                            new_file_face_image, upsampling, 'cnn')
                        # cnnでも顔検出できない場合はnoFaceフォルダへファイルを移動する
                        print(cnt, "Error: 登録顔画像", new_file,
                              "に顔が検出されませんでした(CNNモード)。 noFace フォルダへ移動します")

                        try:
                            shutil.move(new_file, '../noFace/')
                        except:
                            os.remove('../noFace/' + new_file)
                            shutil.move(new_file, '../noFace/')

                        mode = 'hog'
                        print('hogモードへ戻しました')

                # new_file顔画像のエンコーディング処理：array([encoding 配列])
                print(cnt, ' ', new_file, '\t', 'をエンコードしています')
                new_file_face_encodings = face_recognition.face_encodings(
                    new_file_face_image, new_file_face_locations, jitters, 'small')

                if len(new_file_face_encodings) > 1:  # 複数の顔が検出された時
                    print(cnt, "Error: 登録顔画像", new_file,
                          "に複数の顔が検出されました。 noFace フォルダへ移動します")
                    shutil.move(new_file, '../noFace/')
                elif len(new_file_face_encodings) == 0:  # 顔が検出されなかった時
                    print(cnt, "Error: 登録顔画像", new_files,
                          "に顔が検出されませんでした。 noFace フォルダへ移動します")
                    shutil.move(new_file, '../noFace/' + new_file)

                # エンコーディングした顔画像だけ新しい配列に入れる
                known_face_names_list.append(new_file)
                known_face_encodings_list.append(new_file_face_encodings[0])

                cnt += 1

    # ============= npKnown.npzファイルが存在しない場合の処理 =============
    elif not os.path.exists("npKnown.npz"):
        os.chdir(priset_face_imagesDir)
        # まずpriset_face_imagesDirのファイル名を全て得る
        for priset_face_image_filename in os.listdir(priset_face_imagesDir):
            # 関係ないファイルやフォルダは処理からとばす
            if priset_face_image_filename == 'desktop.ini':  # desktop.iniは処理をとばす
                continue
            if os.path.isdir(priset_face_image_filename):  # フォルダの場合は処理をとばす
                continue
            if 'debug' in priset_face_image_filename:  # ファイル名にdebugを含む場合は処理をとばす
                continue

            # それぞれの顔写真について顔認証データを作成する
            pricet_face_img = face_recognition.load_image_file(
                priset_face_image_filename)
            priset_face_img_locations = face_recognition.face_locations(
                pricet_face_img, upsampling, mode)

            # 得られた顔データについて顔写真なのに顔が判別できない場合や複数の顔がある場合はcnnモードで再確認し、それでもな場合はnoFaceフォルダに移動する
            noFace_file = '../noFace/' + priset_face_image_filename
            if len(priset_face_img_locations) == 0 or len(priset_face_img_locations) > 1:
                if mode == 'hog':
                    print('顔が検出できない又は複数検出されました。一時的にcnnモードへ切り替えます')
                    # CNNモードにて顔検出を行う
                    priset_face_img_locations = face_recognition.face_locations(
                        pricet_face_img, upsampling, 'cnn')
                    # cnnでも顔検出できない場合はnoFaceフォルダへファイルを移動する
                    if len(priset_face_img_locations) == 0 or len(priset_face_img_locations) > 1:
                        print(cnt, "Error: (CNNモード)登録顔画像", priset_face_image_filename,
                              "にて顔が検出できない又は複数検出されました。 noFace フォルダへ移動します")
                        if os.path.exists(noFace_file):
                            os.remove(noFace_file)
                        shutil.move(priset_face_image_filename, '../noFace/')
                        mode = 'hog'
                        print('hogモードへ戻しました')

            # 得られた顔データ（この場合は顔ロケーション）を元にエンコーディングする：array([encoding 配列])
            print(cnt, ' ', priset_face_image_filename, '\t', 'をエンコードしています')
            priset_face_image_encodings = face_recognition.face_encodings(
                pricet_face_img, priset_face_img_locations, jitters, 'small')

            # エンコーディングした顔写真について複数顔や顔がない場合はnoFaceフォルダへ移動する
            if len(priset_face_image_encodings) > 1:  # 複数の顔が検出された時
                print(cnt, "Error: 登録顔画像", priset_face_image_filename,
                      "に複数の顔が検出されました。 noFace フォルダへ移動します")
                if os.path.exists(noFace_file):
                    os.remove(noFace_file)
                try:
                    shutil.move(priset_face_image_filename, '../noFace/')
                except:
                    pass
            elif len(priset_face_image_encodings) == 0:  # 顔が検出されなかった時
                print(cnt, "Error: 登録顔画像", priset_face_image_filename,
                      "に顔が検出されませんでした。 noFace フォルダへ移動します")
                if os.path.exists(noFace_file):
                    os.remove(noFace_file)
                try:
                    shutil.move(priset_face_image_filename, '../noFace/')
                except:
                    pass

            # 配列に、名前やエンコーディングデータを要素として追加する
            # FACE01GRAPHICS本体の方では要素にndarrayを含むListを返り値として期待している(face_recognition APIにそう書いてある)
            known_face_names_list.append(priset_face_image_filename)
            known_face_encodings_list.append(priset_face_image_encodings[0])

            cnt += 1

    ###################### np.savezで保存 ######################
    os.chdir(kaoninshoDir)
    # print('debug_npKnown.npzを作成します')
    np.savez(
        'npKnown',
        # known_face_names_list_127=known_face_names_list,
        known_face_names_list,
        # known_face_encodings_list_127=known_face_encodings_list
        known_face_encodings_list
    )

    # ################### リスト型を返す ###################
    # <DEBUG>

    # list=[]
    # for i in known_face_encodings:  ## shape:(677, 1, 128)
    #     for x in i:
    #         list.append(x)
    # known_face_encodings = list
    return known_face_encodings_list, known_face_names_list

    # #################### 備考 ####################
    # 返り値のknown_face_encodingsと、npKnown.npzから読み込んだknown_face_encodingsとでは
    # もしかしたらデータ型とか？なにかが異なっているのかもしれない。


# 検証用
if __name__ == '__main__':
    import pprint
    import sys
    pprint(sys.path)
    exit()

    # <DEBUG>
    kaoninshoDir = '/home/terms/ビデオ/one_drive/FACE01GRAPHICS123_UBUNTU_VENV'
    os.chdir(kaoninshoDir)
    known_face_encodings, known_face_names = load_priset_image(
        kaoninshoDir,
        priset_face_imagesDir=kaoninshoDir + '/priset_face_images',
        upsampling=0,
        jitters=4,
        mode='cnn',
        model='small'
    )
    print('known_face_encodings: ', known_face_encodings)
    print('known_face_names: ', known_face_names)
