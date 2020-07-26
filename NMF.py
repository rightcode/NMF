import numpy as np
import os
import torch
import cv2
from torchnmf import NMF
from PIL import Image
import random

def preserve_Reconstruct_img_addition(params,Q_image,img_path):
  y = 0
  for i in range(Q_image.shape[0]):
    if type(params[i]) is torch.Tensor:
      params[i] = params[i].detach().numpy()
    y += Q_image[i]*params[i]*255
  pil_img = Image.fromarray(y)
  pil_img.convert("L").save(img_path,"JPEG")

def train(params,Q_image,image):
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

def give_Random_noize_to_img(image,rate):
  for width in range(image.shape[0]):
    for height in range(image.shape[1]):
      if random.random() > rate:
        image[width][height] = 0
  return image

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
  Q = model.H.detach().cpu().numpy()

  Q_image = Q.reshape(n_components,resize[0],resize[1])
  img_vec = img_vec.reshape(size,resize[0],resize[1])

  path = "./drive/My Drive/face/tmp/3"
  file = os.listdir(path)
  image = cv2.resize(cv2.cvtColor(np.array(Image.open(path+"/"+file[1])), cv2.COLOR_BGR2GRAY),resize)/255
  image = give_Random_noize_to_img(image,0.8)

  pil_img = Image.fromarray(image*255)
  pil_img.convert("L").save("./drive/My Drive/result_NMF/results_noize.jpg","JPEG")
  params = []
  for i in range(Q_image.shape[0]):
    params.append(torch.zeros(1, requires_grad=True))
  for loop in range(100):
    params = train(params,Q_image,image)
  preserve_Reconstruct_img_addition(params,Q_image,"./drive/My Drive/result_NMF/predict.jpg")
  preserve_Reconstruct_img_addition(P[0],Q_image,"./drive/My Drive/result_NMF/answer.jpg")

if __name__ == '__main__':
  main() 