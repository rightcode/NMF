
# 画像サイズを下げるプログラム
# 大量に画像を読み込むときに元画像が大きいと3000枚程度しか読み込めなかったため

import numpy as np
import os
import cv2
from PIL import Image

path = "./drive/My Drive/man/sub"
file = os.listdir(path)
size = len(file)
resize_w = 200
resize_h = 200 
resize = (resize_w,resize_h)
start = 0
end = size

for i in range(start,end):
  print(i)
  y = cv2.resize(np.array(Image.open(path+"/"+file[i])), resize)
  pil_img = Image.fromarray(y)
  pil_img.convert("L").save(path+"/"+file[i],"JPEG") 