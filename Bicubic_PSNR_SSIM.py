import numpy as np
import cv2
from os.path import join, exists, split
from os import listdir, makedirs
from math import log10
from skimage.metrics import structural_similarity as ssim
import matlab_imresize
import time

# 定义一个用于图像缩放的类
class Resize(object):
    def __init__(self, scale_factor):
        # 初始化时设置缩放因子
        self.scale_factor = scale_factor
    
    def __call__(self, img):
        # 计算图像的新尺寸
        dsize = (int(np.shape(img)[1] / self.scale_factor), int(np.shape(img)[0] / self.scale_factor))
        # 使用 matlab_imresize 库进行图像缩放
        return matlab_imresize.imresize(img, output_shape=dsize)

# 定义一个计算峰值信噪比（PSNR）的函数
def get_psnr(img1, img2, min_value=0, max_value=255):
    mse = np.mean((img1 - img2) ** 2)  # 计算均方误差
    if mse == 0:
        return 100  # 如果两张图像完全相同，返回一个较大的数值
    PIXEL_MAX = max_value - min_value
    return 10 * log10((PIXEL_MAX ** 2) / mse)  # 使用PSNR公式计算

# 定义一个用OpenCV保存图像的函数
def save_image(img, path):
    success = cv2.imwrite(path, img)  # 将图像保存到指定路径
    if not success:
        print(f"无法保存图像到 {path}")
    return success  # 返回保存是否成功的状态

# 设置原始测试图像目录和结果图像保存目录
origin_test_dir = r'E:\OneDrive\WorkSpace\SR\Dataset\GIR50\val'
result_dir = r'E:\OneDrive\WorkSpace\SR\Other\Bicubic\result\x4\GIR50'
if not exists(result_dir):
    makedirs(result_dir)  # 如果不存在结果目录，则创建

# 获取所有测试图像的完整路径列表
origin_tests = [join(origin_test_dir, x) for x in sorted(listdir(origin_test_dir))]

# 初始化图像缩放对象，指定超分辨率倍率
resize = Resize(4)  # 4倍下采样
resize_inverse = Resize(1 / 4)  # 将图像还原到原始分辨率

# 初始化列表，用于存储PSNR、SSIM值和处理时间
psnrs = []
ssims = []
times = []

# 遍历每一张原始测试图像
for origin_test in origin_tests:
    print(f"正在处理文件: {origin_test}")
    img_hr = cv2.imread(origin_test, cv2.IMREAD_COLOR)  # 以彩色模式读取高分辨率图像

    if img_hr is None:
        print(f"无法读取图像: {origin_test}")
        continue

    # 将图像从BGR颜色空间转换到YCrCb颜色空间
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2YCrCb)
    img_hr = img_hr.astype(np.float64)  # 转换数据类型为float64以确保计算精度

    # 开始计时
    start_time = time.time()

    # 对图像进行下采样和上采样过程
    img_lr = resize(img_hr)  # 下采样
    img_hr_new = resize_inverse(img_lr)  # 上采样回原始尺寸

    # 仅使用Y通道进行PSNR和SSIM计算
    img_hr_y = img_hr[4:-4, 4:-4, 0]  # 对图像进行裁剪以避免边界效应
    img_hr_new_y = img_hr_new[4:-4, 4:-4, 0]

    # 将像素值裁减到有效范围并转换为uint8类型
    img_hr_y = img_hr_y.clip(0, 255).astype(np.uint8)
    img_hr_new_y = img_hr_new_y.clip(0, 255).astype(np.uint8)

    # 计算PSNR值
    psnr = get_psnr(img_hr_y.astype(np.float64), img_hr_new_y.astype(np.float64), 0, 255)
    psnrs.append(psnr)  # 存储PSNR值

    # 计算SSIM值
    ssim_value = ssim(img_hr_y, img_hr_new_y, data_range=img_hr_new_y.max() - img_hr_new_y.min())
    ssims.append(ssim_value)  # 存储SSIM值

    # 记录处理结束时间并计算处理所用时间
    end_time = time.time()
    times.append(end_time - start_time)  # 存储处理时间

    # 提取文件名并准备保存路径
    filename = split(origin_test)[-1]  # 提取文件名
    result_filename = join(result_dir, f"SR_{filename}")  # 构建保存路径
    
    # 将图像从YCrCb颜色空间转换回BGR
    img_hr_new_bgr = cv2.cvtColor(img_hr_new.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    
    # 保存处理后的图像到指定路径
    if save_image(img_hr_new_bgr, result_filename):
        print(f"图像成功保存: {result_filename}")
    else:
        print(f"图像保存失败: {result_filename}")

# 输出所有图像的平均PSNR, SSIM值和平均处理时间
print('平均 PSNR:', np.mean(psnrs))
print('平均 SSIM:', np.mean(ssims))
print('平均处理时间:', np.mean(times))
