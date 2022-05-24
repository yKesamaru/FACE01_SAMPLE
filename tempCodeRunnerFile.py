def draw_text_for_name(left,right,bottom,name, p,tolerance,pil_img_obj):
    fontpath = return_fontpath()
    """TODO FONTSIZEハードコーティング訂正"""
    fontsize = 14
    font = ImageFont.truetype(fontpath, fontsize, encoding = 'utf-8')
    # テキスト表示位置決定
    position, Unknown_position = calculate_text_position(left,right,name,fontsize,bottom)
    # nameの描画
    pil_img_obj = draw_name(name,pil_img_obj, Unknown_position, font, p, tolerance, position)
    # pil_img_objをnumpy配列に変換
    small_frame = convert_pil_img_to_ndarray(pil_img_obj)
    return small_frame
