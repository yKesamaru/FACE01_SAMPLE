from face01lib.example import example

# example("DEBUG", "0", 750)  # USB CAM 未実装
example("DEBUG", "some_people.mp4", 750)
# example("DEBUG", "顔無し区間を含んだテスト動画.mp4", 750)
# example("DEBUG", "http://101.235.184.86:8000/webcapture.jpg?command=snap&channel=1", 1200)  # 床屋
# example("DEBUG", "http://175.210.52.167:84/SnapshotJPEG?Resolution=640x480", 1200)  # 学習塾  
# example("DEBUG", "http://153.220.3.203:80/SnapshotJPEG?Resolution=1280", 750)  # 事務所
# example("DEBUG", "http://128.53.158.70:82/cgi-bin/camera?resolution=1280", 750)  # オフィス