# config.ini
You can configure FACE01 using the `config.ini` file.
Format is `key = value`.
## [DEFAULT]
NOTE: Do not edit.
- model
  - str
  - Default: small
  - Do *not* override.

## [Main]
- headless
  - bool
  - Default: True
  - If set `False`, display GUI window, and process speed get slowly.
- anti_spoof
  - **Experimental**
  - NOTE: Anti-spoof model is included with this sample, but please do ***not*** use this model as is for commercial use. Please contact tokai-kaoninsho for details.
  - bool
  - Default: False

## [SPEED OR PRECISE]
- set_width
  - int
  - Default: 750
  - Width of GUI window.
- similar_percentage
  - float
  - Default: 99.1
- jitters
  - int
  - Default: 0
  - Calculate jitters on running FACE01.
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

##  [dlib]
- use_pipe
  - bool
  - Default: True
  - Whether to use dlib for face coordinate calculation.
- model_selection
  - int
  - Default: 1
  - O OR 1,0: Within 2 meters from the camera, 1: Within 5 meters. More than that, specify use_pipe = False.
  - The minimum confidence value from the face detection model for the detection to be considered successful. If wearing the mask, set it to about 0.3. The lower the number, the higher the possibility of erroneous recognition other than the face. The standard is 0.4 to 0.5.
  - NOTE: This value is set only when `use_pipe` is `True`.
- min_detection_confidence
  - float
  - Default: 0.4
  - NOTE: You can set `person_frame_face_encoding` to `True` only if `use_pipe` is True.
- person_frame_face_encoding
  - bool
  - Default: False
  - You can set person_frame_face_encoding to True only if `use_pipe` is `True`.
- same_time_recognize
  - int
  - Default: 2
  - Number of people to recognize at the same time. Default is 2. Valid only if `use_pipe` is `True`.

##  [INPUT]
- set_area
  - str
  - Default: NONE
  - Zoom. You can select from `NONE`, `TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`, `CENTER`.
- movie
  - str
  - Default: some_people.mp4
  - For test, you can select from bellow.
    - usb (or USB)
      - USB Cam
    - test.mp4 (Only a person.)
    - 顔無し区間を含んだテスト動画.mp4
      - Movie file which contain no person frames.
    - rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4
      - RTSP stream for test.
    - http://219.102.239.58/cgi-bin/camera?resolution=750
      - Live stream using HTTP for test: Live cam at Tokyo.

## [Authentication]
- user = ""
- passwd = ""

## [DRAW INFORMATION]
- rectangle
  - bool
  - Default: False
  - Display a legacy face frame on the screen
- target_rectangle
  - bool
  - Default: True
  - Display a modern face frame on the screen
  - NOTE: You can select only one of `rectangle` or `target_rectangle`.
- draw_telop_and_logo
  - bool
  - Default: True
  - Display of telop and logo
- default_face_image_draw
  - bool
  - Default: True
  - Display the registered face image on the screen
- show_overlay
  - bool
  - Default: True
  - Make the information display on the screen semi-transparent
- show_percentage
  - bool
  - Default: True
  - Show similarity
- show_name
  - bool
  - Default: True
  - Show name

## [SAVE FACE IMAGE]
- crop_face_image
  - bool
  - Default: True
  - Save face images
- frequency_crop_image
  - int
  - Default: 5
  - Save face images per frame
- crop_with_multithreading
  - bool
  - Default: True
  - Save face images using multi-threading. If using slower storage, set 'True'.

## [system_check]
NOTE: Do not edit.
- Python_version
  - str
  - Default: 3.8.10
- cpu_freq
  - float
  - Default: 2.5
  - GHz
- cpu_count
  - int
  - Default: 4
- memory
  - int
  - Default: 4
  - GByte
- gpu_check
  - bool
  - Default: True

## [DEBUG]
NOTE: Do not edit.
- calculate_time
  - bool
  - Default: True
  - time measurement

## [Scheduled to be abolished]
NOTE: Do not edit.
- show_video
  - bool
  - Default: False