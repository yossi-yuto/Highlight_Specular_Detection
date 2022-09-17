import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb


# RGBから明度の計算 式（１）
def calc_brightness(img_rgb):
  height, width, _ = img_rgb.shape
  assert img_rgb.dtype == np.float64, 'img_rgb.dtype needs np.float64'
  img_rgb_square = np.square(img_rgb)
  bright_map = np.sqrt(0.241 * img_rgb_square[:,:,0] + 0.691 * img_rgb_square[:,:,1] + 0.068 * img_rgb_square[:,:,2])
  return  np.sum(bright_map) / (height * width)


# Algorithm1
def contrast_equalization(img_rgb, threshold_brightness):
  img_rgb = img_rgb.astype(np.float64)
  contrast = 1.
  brightness = calc_brightness(img_rgb)
  if (brightness > threshold_brightness):
    while(brightness > threshold_brightness):
      contrast -= 0.01
      img_rgb *= contrast
      brightness = calc_brightness(img_rgb)
  return img_rgb, brightness

# パスからRGB画像とHSV画像（Saturation,Value）を返す関数
def image_rgb_hsv(img_path, resize=None):
    img = cv2.imread(img_path)
    if resize is not None:
      img = cv2.resize(img, resize)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return rgb_img, hsv_img


# Fig4. データ分布が異なるためおそらく人手に設定しなければならない。
def optimal_value_threshold(brightness):
  return 2. * brightness


# 勾配強度マップ
def image_gradient_M(input_image, image_flag=None):
  '''
  input_image : 画像の配列
  return : 画像の勾配でinput_image(H,W,C)
  '''
  sobelx64f = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=5)
  sobely64f = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=5)

  return np.sqrt(np.square(sobelx64f) + np.square(sobely64f))


def main():
  path = 'data/CIMG7857.jpg'  
  img_rgb, img_hsv = image_rgb_hsv(path)
  S = img_hsv[:,:,1]  # Saturation
  V = img_hsv[:,:,2]  # Value

  '''前処理 Sec(3.2) '''
  img_rgb, brightness = contrast_equalization(img_rgb, threshold_brightness=100)  # 前処理（3.2） 論文では125に設定

  '''閾値処理　Sec(3.3) '''
  T_value = optimal_value_threshold(brightness)  # 最適な閾値を計算
  T_saturation = 10
  specular_img = 1 - np.uint8(np.where(S < T_saturation, 1, 0) == np.where(V > T_value, 1, 0))

  ''' 後処理 Sec(3.4) '''
  grad_intesity_map = image_gradient_M(img_hsv)
  #image = {"original image" : img_rgb, "specular highright image" : specular_img}  
  for T_value in np.arange(230,255):
    area = np.where(V > T_value, 1, 0)
    
  
  '''図の描画'''
  fig, ax = plt.subplots(1,3, figsize=(20,10))
  ax[0].imshow(img_rgb.astype(np.uint8))
  divider = make_axes_locatable(ax[1])
  cax = divider.append_axes('right', size='5%', pad=0.05)
  im = ax[1].imshow(specular_img, cmap='gray')
  ax[2].imshow(grad_intesity_map)
  fig.colorbar(im, cax=cax, orientation='vertical')
  fig.savefig("test.png")

if __name__ == '__main__':
  main()