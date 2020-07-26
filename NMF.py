# !git clone https://github.com/yoyololicon/pytorch-NMF
import numpy as np
import os
import torch
import cv2
from torchnmf import NMF
from PIL import Image

import random
path = "./drive/My Drive/man/sub"
file = os.listdir(path)
size = 3000
resize = (100,100)
n_components = 3000
img_vec = np.zeros((size,resize[0]*resize[1]))
parms_list = [0] * n_components
learning_rate = 0.001
for i in range(size):
  if i %100 == 0:
    print(i)
  img_vec [i,:]= cv2.resize(np.array(Image.open(path+"/"+file[i])), resize).reshape(resize[0]*resize[1])
img_vec = img_vec/255
model = NMF(img_vec.shape, rank=n_components).cuda()
_,reconstruct = model.fit_transform(torch.from_numpy(img_vec).cuda(),l1_ratio=1)
model.sort()
P = model.W.detach().cpu().numpy()
Q = model.H.detach().cpu().numpy()
for number in range(P.shape[0]):
  koyuuti = P[number]
  Q_image = Q.reshape(n_components,resize[0],resize[1])
  img_vec = img_vec.reshape(size,resize[0],resize[1])
  result = np.zeros((resize[0],resize[1]))
params = []
path = "./drive/My Drive/face/tmp/3"
file = os.listdir(path)
image = cv2.cvtColor(np.array(Image.open(path+"/"+file[1])), cv2.COLOR_BGR2GRAY)
for width in range(image.shape[0]):
  for height in range(image.shape[1]):
    if random.random() > 0.8:
      image[width][height] = 0

image = cv2.resize(image, resize)

image = image/255

pil_img = Image.fromarray(image*255)
pil_img.convert("L").save("./drive/My Drive/result_NMF/results_noize.jpg","JPEG")
for i in range(Q_image.shape[0]):
  params.append(torch.zeros(1, requires_grad=True))
for loop in range(100):
  y = 0
  for i in range(Q_image.shape[0]):
    params[i] = torch.tensor(params[i],requires_grad=True)
    y += torch.from_numpy(Q_image[i])*params[i]
  MSE = ((y - torch.from_numpy(image))**2).mean()
  with torch.no_grad():
    MSE.backward()
    if i % 50 == 0:
      print(MSE)
  for i in range(Q_image.shape[0]):
    params[i] = params[i] - params[i].grad*learning_rate
y = 0
for i in range(Q_image.shape[0]):
  y += Q_image[i]*params[i].detach().numpy()*255
pil_img = Image.fromarray(y)
pil_img.convert("L").save("./drive/My Drive/result_NMF/predict.jpg","JPEG")
y = 0
for i in range(Q_image.shape[0]):
  y += Q_image[i]*P[0][i]*255
pil_img = Image.fromarray(y)
pil_img.convert("L").save("./drive/My Drive/result_NMF/answer.jpg","JPEG")