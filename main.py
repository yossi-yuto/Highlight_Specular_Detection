from filecmp import cmp
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

# ゼロ除算対策
def div(grad_, grad_M):
  grad_[grad_M != 0.] /= grad_M[grad_M != 0.]
  return grad_ 
  

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


# Sobel Filter
def image_gradient_with_M(input_image):
  sobelx64f = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=5)
  sobely64f = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=5)
  magunitude = np.sqrt(np.square(sobelx64f) + np.square(sobely64f))
  # 正規化
  norm_x = div(sobelx64f, magunitude)
  norm_y = div(sobely64f, magunitude)
  return norm_x, norm_y, magunitude


# 勾配方位
def image_gradient_O(input_image):
  sobelx64f, sobely64f, _ = image_gradient_with_M(input_image)
  with np.errstate(divide='ignore'): 
    orientation = sobely64f / sobelx64f
  orientation[np.isnan(orientation)] = 0
  orientation = np.arctan(orientation)
  return orientation

def main():
  path = 'data/CIMG7857.jpg'  
  img_rgb, img_hsv = image_rgb_hsv(path)
  S = img_hsv[:,:,1]  # Saturation
  V = img_hsv[:,:,2]  # Value

  img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
  magunitude = image_gradient_O(img_gray)
  plt.imshow(magunitude, vmin= -np.pi / 2, vmax=np.pi / 2, cmap='jet')
  plt.axis('off')
  plt.colorbar()
  plt.savefig("orientation.jpg")

  '''前処理 Sec(3.2) '''
  img_rgb, brightness = contrast_equalization(img_rgb, threshold_brightness=100)  # 前処理（3.2） 論文では125に設定

  '''閾値処理 Sec(3.3) '''
  T_value = optimal_value_threshold(brightness)  # 最適な閾値を計算
  T_saturation = 10
  specular_img = 1 - np.uint8(np.where(S < T_saturation, 1, 0) == np.where(V > T_value, 1, 0))

  ''' 後処理 Sec(3.4) '''
  pdb.set_trace()
  grad_intesity_map = image_gradient_with_M(img_hsv)
  #image = {"original image" : img_rgb, "specular highright image" : specular_img}  
  pixels = [np.sum(np.where(V > T_value, 1, 0)) for T_value in np.arange(230,255)]

  fig, ax = plt.subplots(1)
  ax.plot(np.arange(230,255), pixels)
  fig.savefig('specularity_evolution.jpg')
  pdb.set_trace()
    
  
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