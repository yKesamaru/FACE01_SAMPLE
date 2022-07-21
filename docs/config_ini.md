# config.ini
You can configure FACE01 using the `config.ini` file.
Format is `key = value`.

- headless
  - bool
  - Default: True
    - If `True`, display GUI window.
      - set_width
        - int
        - Default: 750
          - Width of GUI window.
- similar_percentage
  - Default: 99.1
similar_percentage = 99.1
- jitters
  - int
  - Default: 0
  - Calculate jitters for running FACE01.
- priset_face_images_jitters
  - int
  - Default: 100
  - Calculate jitters for priset_face_images.
- upsampling
  - int
  - Default: 0
  - Specifying the detected face area. 0: 80x80px, 1: 40x40px
- mode
  - str
  - Default: cnn
  - If you don't use cuda, set 'hog'.
- frame_skip
  - int
  - Default: 5
  - Specify the number to drop frame. Do not make it less than 2 if use http protocol.
- number_of_people
  - int
  - Default: 10
  - Do not analyze more than the specified number of people.
-
- [dlib]
- Whether to use dlib for face coordinate calculation.
use_pipe = True
- O OR 1,0: Within 2 meters from the camera, 1: Within 5 meters. More than that, specify use_pipe = False.
model_selection = 1
- The minimum confidence value from the face detection model for the detection to be considered successful. If wearing the mask, set it to about 0.3. The lower the number, the higher the possibility of erroneous recognition other than the face. The standard is 0.4 to 0.5.
min_detection_confidence = 0.4
- You can set person_frame_face_encoding to True only if use_pipe == True.
person_frame_face_encoding = False
-
- [INPUT]
- Zoom. [NONE, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER]から選びます。
- set_area = BOTTOM_RIGHT
set_area = NONE
- USB Cam.
- movie = usb
- Movie file for test.
- movie = test.mp4
- movie = 顔無し区間を含んだテスト動画.mp4
movie = some_people.mp4
- RTSP stream for test.
- movie = rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4
- Live stream using HTTP for test: Live cam at Tokyo.
- movie = http://219.102.239.58/cgi-bin/camera?resolution=750
-
- [Authentication]
user = ""
passwd = ""
-
- [DRAW INFOMATION]
- Display a legacy face frame on the screen
rectangle = False
- Display a modern face frame on the screen
target_rectangle = True
- Display of telop and logo
draw_telop_and_logo = True
- Display the registered face image on the screen
default_face_image_draw = True
- Make the information display on the screen semi-transparent
show_overlay = True
- Show similarity
show_percentage = True
- Show name
show_name = True
-
- [SAVE FACE IMAGE]
- Save face images
crop_face_image = True
- Save face images per frame
frequency_crop_image = 5
- Save face images using multi-threading. If using slower strage, specify 'True'.
crop_with_multithreading = True
-
- [system_check]
Python_version = 3.8.10
- GHz
cpu_freq = 2.5
cpu_count = 4
- GByte
memory = 4
gpu_check = True
- [DEBUG]
- time measurement
calculate_time = True
- [Scheduled to be abolished]
show_video = False