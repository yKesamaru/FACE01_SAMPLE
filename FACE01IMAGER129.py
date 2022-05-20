import datetime
import glob
import os
import re
import shutil

import cv2
import face_recognition
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from PySimpleGUI.PySimpleGUI import POPUP_BUTTONS_OK

from face01lib129 import load_priset_image
from face01lib129.similar_percentage_to_tolerance import (to_percentage, to_tolerance)

# version 1.2.9 IMAGER Linux & Windows リリースノート ==================
# 細かいバグフィックス
# Face_attestationクラス化
# ======================================================================

# version 1.2.8 IMAGER Linux & Windows リリースノート ==================
# FACE01 GRAPHICS ver.1.2.8に準拠
# torelance指定からsimilar_percentage指定へ変更
# face01lib128を使用するよう変更
# ======================================================================

# version 1.2.5 IMAGER Linux & Windows リリースノート ==================
# FACE01 GRAPHICS ver.1.2.5に準拠
# コメント削除→1.1.7を参照
# ======================================================================

# version 1.1.7 imager Linux & Windows リリースノート ==================
# 各機能のモジュール化
# ======================================================================


# 指定日付計算 =========================================================
limmit_date = datetime.datetime(2022, 12, 1, 0,0,0)  ## 指定日付
today = datetime.datetime.now()
sg.theme('LightGray')
def limmit_date_alart():
    if today >= limmit_date:
        print('指定日付を過ぎました')
        sg.popup( 'サンプルアプリケーションをお使いいただきありがとうございます','使用可能期限を過ぎました', '引き続きご利用になる場合は下記までご連絡下さい', '東海顔認証　担当：袈裟丸','y.kesamaru@tokai-kaoninsho.com', '', 'アプリケーションを終了します', title='', button_type=POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
        exit()
    elif today < limmit_date:
        remmaining_days = limmit_date - today
        if remmaining_days.days < 30:
            dialog_text = 'お使い頂ける残日数は' + str(remmaining_days.days) + '日です'
            sg.popup( 'サンプルアプリケーションをお使いいただきありがとうございます', dialog_text, title='', button_type=POPUP_BUTTONS_OK, modal=True, keep_on_top=True)
limmit_date_alart()

# ホームディレクトリ固定 ===============================================
def home():
	kaoninshoDir = os.getcwd()
	priset_face_imagesDir = kaoninshoDir + '/priset_face_images/'
	check_images = kaoninshoDir + '/check_images/'
	return kaoninshoDir, priset_face_imagesDir, check_images  ## 複数の戻り値はタプルで返される
# ======================================================================

# 画像認識部 ===========================================================
class Face_attestation:
	def __init__(
			self,
			check_images,
			known_face_encodings,
			known_face_names,
			jitters,
			similar_percentage,
			mode,
			upsampling
		):
		self.check_images=check_images
		self.known_face_encodings=known_face_encodings
		self.known_face_names=known_face_names
		self.jitters=jitters
		self.similar_percentage=similar_percentage
		self.mode=mode
		self.upsampling=upsampling

	def face_attestation(
			kaoninshoDir,
			check_images, 
			known_face_encodings, 
			known_face_names, 
			similar_percentage=99.0,
			jitters=0,
			upsampling=0,
			mode='hog',
		):

		# 変数初期化 =============
		check_images_files=[]
		# ========================

		# toleranceの算出 ========
		tolerance = to_tolerance(similar_percentage)
		# ========================

		os.chdir(check_images)

		# check_images_files=glob.glob("*.[jJ][pP]*")  ## 大文字小文字
		files=glob.glob('*')

		# フォルダ内のjpeg,png画像をリストに格納
		check_images_files = [file for file in files 
						if re.search('.*(jpeg|jpg|png)', file, re.IGNORECASE)]

		for check_images_file in check_images_files:

			face_names = []

			# 顔画像ファイルを読み込む
			check_images_file_npData = cv2.imread(check_images_file)
			
			# <要修正> -----------
			if check_images_file_npData is None:
				print(check_images_file, ' :読込み不能です')
				continue
			# --------------------
			
			# BGRからRGBへ変換
			check_images_file_npData = check_images_file_npData[:, :, ::-1]

			# face_locationsを算出
			face_locations = face_recognition.face_locations(check_images_file_npData, upsampling, mode)

			# 顔が1つも見つからなかった場合
			if len(face_locations) == 0:  
				
				# hog, cnnモードの自動切り替えブロック
				if mode == 'hog':  # modeがhogだった場合
					print(check_images_file, 'に顔を検出できませんでした。',
					'mode を hog から cnn にして再試行...')
					mode='cnn'
					print("mode = 'cnn' で顔探索中…")
					face_locations = face_recognition.face_locations(check_images_file_npData, upsampling, mode)
					
					if len(face_locations) == 0:
						print(check_images_file, ' には mode=cnn でも顔は検出できませんでした。', 
						check_images_file, ' を noFace フォルダに移動します。')
						shutil.move(check_images_file, "../noFace/" + check_images_file)
						mode = 'hog'
						continue
					else:
						print("顔を検出しました mode='hog' に戻します \n...OK")
						mode = 'hog'
				
				elif mode == 'cnn':  # modeがcnnの場合
					print('check_images フォルダ内の', check_images_file, 'に顔を検出できませんでした。mode が cnn であるため再試行せず', check_images_file, ' を noFace フォルダに移動します。')
					shutil.move(check_images_file, "../noFace/" + check_images_file)
					continue

				else:  # modeがhogでもcnnでも無かった場合
					print('オプション変数modeにはhogかcnnを指定してください')
					print('終了します')
					exit()
				
				# 複数の顔が検出された場合
				if len(face_locations) > 1:
					print(f'{check_images_file} に複数の顔が見つかりました')
					print('noFaceフォルダに移動します')
					shutil.move(check_images_file, "../noFace/" + check_images_file)
					continue
			
			# 顔をエンコーディングする
			face_encodings = face_recognition.face_encodings(check_images_file_npData, face_locations, jitters, model='small')
			
			# 与えられたface_encodingsについて順次処理後、face_namesリストに格納する
			for face_encoding in face_encodings:

				name = "Unknown"
				
				# tolerance ( 閾値 ) の範囲内で matche する登録顔画像ファイルを true と false で表現する
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)
				
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				
				best_match_index = np.argmin(face_distances)

				if matches[best_match_index]:
					name = known_face_names[best_match_index]
					name, _ = name.split('_', maxsplit=1)

				# name に distance をくっつける ================================
				# nameとdistanceを対にした辞書型を作るより簡単だから。
				distance = round(min(face_distances),2)
				name = name + ':' + str(distance)
				# =============================================================
				
				face_names.append(name)
				
			for (top, right, bottom, left), face_name in zip(face_locations, face_names):

				# face_nameとdistanceを導出
				face_name, distance = face_name.split(':')

				# 戻り値を返し顔画像ファイルをクロップして保存する
				if face_name == 'Unknown':  # face_nameがUnknownだった場合

					# 日付時刻を算出
					date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒
					
					img = Image.fromarray(check_images_file_npData)
					
					imgCroped = img.crop((left -50,top -50,right +50,bottom +50)).resize((200, 200))

					# 名前がUnknownなのでpercentageはNone
					percentage = None
					
					filename = "../output/%s_%s_%s.jpg" % (face_name, percentage, date)
					imgCroped.save(filename)

					# 検証済みの画像を移動する
					shutil.move(check_images_file, "../recognated/" + check_images_file)

					# # カレントディレクトリを元に戻す
					# if os.getcwd()==check_images:
					# 	os.chdir(kaoninshoDir)

					yield{'name':face_name, 'date':date, 'percentage':percentage, 'original_photo':check_images_file}

				else:  # face_nameが既知の名前だった場合

					date = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f") # %f-> マイクロ秒
					
					img = Image.fromarray(check_images_file_npData)
					
					imgCroped = img.crop((left -50,top -50,right +50,bottom +50)).resize((200, 200))

					# distanceを百分率に変換
					percentage = to_percentage(float(distance))
					# str型に型変換
					percentage = str(round(percentage, 2))

					filename = "../output/%s_%s_%s.jpg" % (face_name, percentage + '%', date)
					imgCroped.save(filename)
					
					# 検証済みの画像を移動する
					shutil.move(check_images_file, "../recognated/" + check_images_file)

					# # カレントディレクトリを元に戻す
					# if os.getcwd()==check_images:
					# 	os.chdir(kaoninshoDir)

					yield{'name':face_name, 'date':date, 'percentage':percentage + '%', 'original_photo':check_images_file}
				
			# 検証済みの画像を移動する
			# shutil.move(check_images_file, "../recognated/" + check_images_file)

			# カレントディレクトリを元に戻す
			os.chdir(kaoninshoDir)
		

	def __del__(self):
		print("インスタンスを破棄しました")
# ======================================================================


# main =================================================================
if __name__ == '__main__':

	# 変数設定 ==========================
	## 説明(default値)
	
	## 類似度
	similar_percentage=99.0
	## ゆらぎ値(0)
	jitters=0
	## 登録顔画像のゆらぎ値(100)
	priset_face_images_jitters=100
	## 最小顔検出範囲(0)
	upsampling=0
	## 顔検出方式(hog)
	mode='hog'

	# ===================================
	
	kaoninshoDir, priset_face_imagesDir, check_images = home()

	known_face_encodings, known_face_names = load_priset_image.load_priset_image(
		kaoninshoDir,
		priset_face_imagesDir, 
		jitters=priset_face_images_jitters
	)

	from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

	# pool = ProcessPoolExecutor()
	pool = ThreadPoolExecutor()

	def multi(x):
		name, date, percentage, original_photo = x['name'], x['date'], x['percentage'], x['original_photo']
		print(
			'name', name,
			'date', date,
			'percentage', percentage,
			'original photo', original_photo
		)

	con = Face_attestation

	while(1):
		xs = con.face_attestation(
			kaoninshoDir,
			check_images, 
			known_face_encodings, 
			known_face_names, 
			similar_percentage=similar_percentage,
			jitters=jitters,
			upsampling=upsampling,
			mode=mode
		)

		for x in xs:
			# name, date, percentage, original_photo = x['name'], x['date'], x['percentage'], x['original_photo']
			# print(
			# 	'name', name,
			# 	'date', date,
			# 	'percentage', percentage,
			# 	'original photo', original_photo
			# )
			multi(x)
