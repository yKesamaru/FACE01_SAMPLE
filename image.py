import cv2

# purin.jpegを読み込む
purin = cv2.imread("/home/terms/ダウンロード/purin.jpeg")

# purin.jpegを表示
cv2.imshow("WAKANA", purin)
cv2.waitKey(5000)
cv2.destroyAllWindows()

