# ############################################################
# name_encoding_dict.py
# known_face_encodings_listとknown_face_names_listから辞書ファイルを
# 作りreturnする
# ############################################################
# 
# last update:
# 2021年10月12日
# 2021年10月8日

import itertools
import subprocess

import numpy as np
import pandas as pd
import similar_percentage_to_tolerance
from load_priset_image import load_priset_image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor



def make_dict(kaoninshoDir):
    known_face_encodings_list, known_face_names_list = load_priset_image(
        kaoninshoDir,
        priset_face_imagesDir=kaoninshoDir + '/priset_face_images', 
        upsampling=0, 
        jitters=1, 
        mode='cnn', 
        model='small'
    )

    name_encoding_dict = {}

    # 辞書へ要素を追加
    for counter in range(0, len(known_face_names_list), 1):
        name_encoding_dict[known_face_names_list[counter]] = known_face_encodings_list[counter]

    return name_encoding_dict

def make_pair_and_save(name_encoding_dict):
    pair_distance_dict = {}  # pair_distance_dict -> pair : distance
    distance_list = []

    # 名前のペア作成
    print('名前のペア作成開始')
    pairs_list = list(itertools.combinations(name_encoding_dict.keys(), 2))

    # # dataframe作成→メモリ不足で落ちる
    # print('DataFrame格納開始')
    # df = pd.DataFrame(pairs_list, dtype='unicode')
    # print(df.head())

    # <DEBUG>
    # line_counter = 1
    # for line in pairs_list:
    #     print(line)
    #     if line_counter==5:
    #         break
    #     line_counter=line_counter + 1

    # # ペアのリストをディスクに書き込む
    print('pairs_list.csvに書き込み開始')
    line_counter = 0
    for line in pairs_list:
        with open('pairs_list.csv', mode='a') as fo:
            fo.write(line[0])
            fo.write(',')
            fo.write(line[1])
            fo.write('\n')
        line_counter=line_counter + 1

    print(line_counter, '行書き込みました')

    # # メモリを専有して最後に落ちた方法
    # print('ペアをpairs_list.datとして保存します')
    # with open('pairs_list.dat', mode='wb') as fo:
    #     pickle.dump(pairs_list, fo)
    
    print('正常に終了しました')

# def compute_distance(first, second):
#     distance = np.linalg.norm(name_encoding_dict[first] - name_encoding_dict[second], ord=2, axis=None)
#     yield distance

def hoge(name_encoding_dict):
    fi = open('pairs_list.csv', 'r')
    line = fi.readline()

    line_counter=0
    while line:
        first, second = line.split(',', 1)
        second, _ = second.split('\n')
        # <DEBUG> ----
        # for i in range(0,10):
        #     if i > 5:
        #         exit()
            # print(first, second)
            # print(name_encoding_dict[first])
            # i=i+1
        # ------------

        # <DEBUG> ----
        # print(
        #     first,
        #     second,
        #     np.linalg.norm(name_encoding_dict[first] - name_encoding_dict[second], ord=2, axis=None)
        # )
        # ------------

        # pool = ThreadPoolExecutor()
        # distance = pool.submit(compute_distance, first, second)
        # face_distance = distance.result()

        face_distance = np.linalg.norm(name_encoding_dict[first] - name_encoding_dict[second], ord=2, axis=None)
        if face_distance < 0.32:
            # print(first, second, face_distance); exit()
            # fo = open('pairs_and_distans.csv', 'a')
            # fo.write(first)
            # fo.write(',')
            # fo.write(second)
            # fo.write(',')
            # fo.write(str(face_distance))
            # fo.write('\n')
            # fo.close

            first_name, _ = first.split('_', 1)
            second_name, _ = second.split('_', 1)

            if not first_name==second_name:
                
                with open('pairs_and_distans.csv', 'a') as fo:
                    fo.write(first)
                    fo.write(',')
                    fo.write(second)
                    fo.write(',')
                    fo.write(str(face_distance))
                    fo.write('\n')

            # <DEBUG> ----
            # if line_counter > 5:
            #     exit()
            # line_counter+=1
            # ------------
        else:
            print(first, second, '\t', round(face_distance, 2))
            # continue
        

        line = fi.readline()  # 次の行に移動する
    fi.close
    
    args = ['play' '-v' '0.5' '-q' '/home/terms/done.wav']
    subprocess.run(args)

def tmp():
    print(
        len(list(itertools.combinations(name_encoding_dict.keys(), 2))),
        'の組み合わせから探します'
    )
    for pair in list(itertools.combinations(name_encoding_dict.keys(), 2)):
        distance = np.linalg.norm(name_encoding_dict[pair[0]] - name_encoding_dict[pair[1]], ord=2, axis=None)
        
        pair_distance_dict[pair] = distance
        
        distance_list.append(distance)

    distance_list.sort(reverse=False)

    cnt = 0
    for distance_item in distance_list:
        
        for key, value in pair_distance_dict.items():
            if value==distance_item:

                first, second = key[0], key[1]

                first, _first = first.split('_', 1)
                second, _second = second.split('_', 1)

                if not first==second:
                # if first==second:
                    first = first + '_' + _first
                    second = second + '_' + _second

                    print(
                        cnt + 1, '位',
                        first,
                        second,
                        round(similar_percentage_to_tolerance.to_percentage(value), 2),'%'
                    )
                    cnt+=1
                    if cnt==10:
                        exit()
        
# 検証
if __name__ == '__main__':
    name_encoding_dict = make_dict('/home/terms/ビデオ/one_drive/FACE01GRAPHICS123_UBUNTU_VENV')
    # make_pair_and_save(name_encoding_dict)
    hoge(name_encoding_dict)