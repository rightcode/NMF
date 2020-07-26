import numpy as np
import os
import torch
import cv2
from torchnmf import NMF
from PIL import Image
import random

# 得られたパラメータと基底から画像を近似した画像を返す関数
def preserve_Reconstruct_img_addition(params,Q_image,img_path):
  y = 0
  for i in range(Q_image.shape[0]):
    if type(params[i]) is torch.Tensor:
      params[i] = params[i].detach().numpy()
    y += Q_image[i]*params[i]*255
  pil_img = Image.fromarray(y)
  pil_img.convert("L").save(img_path,"JPEG")

# 基底と目標画像から適切なパラメータを推定する関数
def train(params,Q_image,image,learning_rate):
  y = 0
  for i in range(Q_image.shape[0]):
    params[i] = torch.tensor(params[i],requires_grad=True)
    y += torch.from_numpy(Q_image[i])*params[i]
  MSE = ((y - torch.from_numpy(image))**2).mean()
  with torch.no_grad():
    MSE.backward()
  for i in range(Q_image.shape[0]):
    params[i] = params[i] - params[i].grad*learning_rate
  return params

# 画像にランダムなノイズを与える関数
def give_Random_noize_to_img(image,rate):
  for width in range(image.shape[0]):
    for height in range(image.shape[1]):
      if random.random() > rate:
        image[width][height] = 0
  return image

# 画像を一次元に変換したものをデータ数分まとめた行列を返す関数
def matrix_of_one_dimensional_Images(size,path,file,resize):
  img_vec = np.zeros((size,resize[0]*resize[1]))
  for i in range(size):
    img_vec [i,:]= cv2.resize(np.array(Image.open(path+"/"+file[i])), resize).reshape(resize[0]*resize[1])/255
  return img_vec

def main():
  path = "./drive/My Drive/man/sub"
  file = os.listdir(path)
  size = 3000
  resize = (100,100)
  n_components = 500
  learning_rate = 0.001
  img_vec = matrix_of_one_dimensional_Images(size,path,file,resize)
  model = NMF(img_vec.shape, rank=n_components).cuda()
  _,reconstruct = model.fit_transform(torch.from_numpy(img_vec).cuda(),l1_ratio=1)

  P = model.W.detach().cpu().numpy()
  Q = model.H.detach().cpu().numpy() #NMFにより得られた基底
  
  Q_image = Q.reshape(n_components,resize[0],resize[1]) #基底を二次元画像に変換
  img_vec = img_vec.reshape(size,resize[0],resize[1])
  # preserve_Reconstruct_img_addition(P[index],Q_image,"./drive/My Drive/result_NMF/answer.jpg") #NMFにより復元された画像(indexを適当な値に変える)

  path = "./drive/My Drive/face/tmp/3"
  file = os.listdir(path)
  image = cv2.resize(cv2.cvtColor(np.array(Image.open(path+"/"+file[1])), cv2.COLOR_BGR2GRAY),resize)/255
  image = give_Random_noize_to_img(image,0.8)

  params = []
  #パラメータの初期化（最初は全部0）
  for i in range(Q_image.shape[0]):
    params.append(torch.zeros(1, requires_grad=True))

  #100回パラメータを学習
  for loop in range(100):
    params = train(params,Q_image,image,learning_rate)

  # pil_img = Image.fromarray(image*255)
  # pil_img.convert("L").save("./drive/My Drive/result_NMF/results_noize.jpg","JPEG") #ノイズ画像の保存

  # preserve_Reconstruct_img_addition(params,Q_image,"./drive/My Drive/result_NMF/predict.jpg") #ノイズ除去後の画像

if __name__ == '__main__':
  main() 