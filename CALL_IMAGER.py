import FACE01IMAGER128 as fi

# 変数設定 ==========================
## 説明(default値)

## 類似度(99.0)
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

kaoninshoDir, priset_face_imagesDir, check_images = fi.home()

known_face_encodings, known_face_names = fi.load_priset_image.load_priset_image(
    kaoninshoDir,
    priset_face_imagesDir, 
    jitters=priset_face_images_jitters
)

while(1):
    xs = fi.face_attestation( 
        check_images, 
        known_face_encodings, 
        known_face_names, 
        similar_percentage=similar_percentage,
        jitters=jitters,
        upsampling=upsampling,
        mode=mode
    )

    for x in xs:
        name, date, percentage, original_photo = x['name'], x['date'], x['percentage'], x['original_photo']
        print(
        	'name', name,
        	'date', date,
        	'percentage', percentage,
        	'original photo', original_photo
        )