from matplotlib import pyplot as plt
import cv2
import numpy as np
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import re

# 安装 https://github.com/UB-Mannheim/tesseract/wiki
# tesseract-ocr-w64-setup-5.4.0.20240606.exe （64 位）
# 然后设置为自己的文件位置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 寻找符合条件的矩形
def find_contours(gray, img, min_area=100):
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测（使用Canny算法）
    edges = cv2.Canny(blurred, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓，找到可能的比例尺区域
    for contour in contours:
        # 计算轮廓的周长并进行多边形逼近
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.intp(box)  # 将坐标转换为整数
        # approx = box
        # 如果轮廓是一个四边形，并且面积大于设定的最小值，我们假设它可能是比例尺
        if len(approx) == 4:
            # 计算轮廓的面积
            area = cv2.contourArea(approx)
            # print(area)
            if area < min_area:
                continue  # 跳过小面积的四边形

            # 创建一个新的图形窗口
            plt.figure()
            # 将图像从BGR转换为RGB（因为matplotlib使用的是RGB格式）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 绘制图像
            plt.imshow(img_rgb)

            # 提取比例尺的四个顶点
            scale_box = approx.reshape(4, 2)

            # 使用matplotlib的plot绘制多边形轮廓
            polygon = np.vstack([scale_box, scale_box[0]])  # 回到第一个顶点以闭合轮廓
            plt.plot(polygon[:, 0], polygon[:, 1], color='g', linewidth=5)
            # 展示图像
            plt.show()

            # 按照 y 坐标排序，如果 y 坐标相同，则按 x 坐标排序
            sorted_box = scale_box[np.lexsort((scale_box[:, 0], scale_box[:, 1]))]
            return scale_box


# 按照矩形裁剪图像
def crop_area(image, scale_box):
    # 计算四边形的最小外接矩形
    x, y, w, h = cv2.boundingRect(scale_box)

    # 裁剪出图像中对应的区域
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image


# 识别裁剪区域中的比例尺的文字，不带单位。单位默认为微米
def find_scale_value(cropped_image2):
    # 转为灰度图

    gray = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # gray = cv2.equalizeHist(gray)
    # 使用自适应阈值化处理灰度图像
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 使用形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 调整裁剪比例尺区域（视情况调整裁剪的比例）
    height, width = morph.shape

    # 显示裁剪后的ROI
    plt.imshow(morph, cmap='gray')
    plt.title('Processed ROI')
    plt.show()

    # 尝试不同的OCR配置，改变psm模式
    for psm_mode in [6, 7, 8]:
        # print(f"尝试 PSM 模式: {psm_mode}")
        custom_config = f'--oem 3 --psm {psm_mode}'

        # 使用OCR提取比例尺上的文本信息
        scale_text = pytesseract.image_to_string(morph, config=custom_config)




        # 输出OCR识别结果
        # print(f"OCR识别结果（PSM {psm_mode}）: {scale_text}")

        # 提取比例尺中的数字
        scale_value = re.search(r'\d+', scale_text)
        if scale_value:
            scale_value = float(scale_value.group(0))
            print(f"提取到的比例尺数值: {scale_value} µm")

            return scale_value, morph
            # break
        else:
            print(f"未提取到有效的比例尺数值 (PSM {psm_mode})")
            return None, morph

#根据文件夹名称获取比例尺
def convert_folder_name(folder_name):
    # 正则表达式匹配尺寸部分
    match = re.search(r'_(\d+)_(\d+)(um|mm)', folder_name)

    if match:
        value1 = int(match.group(1))
        value2 = int(match.group(2))
        unit = match.group(3)

        if unit == 'um':
            result = (value1 + value2 / 10) * 0.01  # 转换为米
        elif unit == 'mm':
            result = (value1 + value2) * 1e-3  # 转 换为米
        else:
            result = None

        return round(result, 5)  # 保留三位小数
    else:
        return 0.001


# 参数配置
def Pixel_to_ratio(path,pixel_to_mm_ratio = 0.0001,min_area = 500,Scale_Value_user=500,use_user_pixel_to_ratio=False):
    # 设定四边形的最小面积阈值，设置最小面积，防止提取到太小的颗粒
    # pixel_to_mm_ratio默认为1：500微米，在识别出错时使用 ，只能做以防万一使用
    #Scale_Value_user =500 默认数字为500微米
    #min_area 最小面积区域，防止提取到其他非矿物矩形

    # 读取图像
    #若使用用户自定义的比例尺则不进行下面的操作了
    #比例尺为1像素：n毫米
    if use_user_pixel_to_ratio:
        return pixel_to_mm_ratio
    image = cv2.imread(path)

    if image is None:
        print("Pixel_to_ratio-图片加载失败")
        return None
    else:

        # 获取图像尺寸并裁剪右下角的部分
        h, w = image.shape[:2]
        # min_area = (h/10)*(w/5)
        crop_fraction = 0.8  # 裁剪比例 ,此值越大，裁剪的图像越小，（比例尺越大，越容易观察）
        cropped_image1 = image[int(h * crop_fraction):, int(w * crop_fraction):]
        # 转为灰度图
        gray = cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray)
        # 转换为灰度图像

        # # 应用阈值将比例尺区域与背景分开
        # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        #
        # # 使用形态学操作去除噪声（如闭操作）
        # kernel = np.ones((5, 5), np.uint8)
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        scale_box1 = find_contours(gray, cropped_image1, min_area)

        if scale_box1 is not None:
            # 得到了 scale_box, 它是四个顶点坐标
            cropped_image2 = crop_area(cropped_image1, scale_box1)

            # 识别比例尺文字
            scale_value, morph = find_scale_value(cropped_image2)

            #####################################################################################################
            #####################################################################################################

            # 若识别失败，则放缩或更换cropped_image1的裁剪区域（四个角），来检查其他位置是否有比例尺内容

            #####################################################################################################
            #####################################################################################################

            # 转为灰度图
            scale_box2 = find_contours(morph, morph)
            if scale_value is None:
                if scale_box2 is None:
                   pixel_to_mm_ratio = (Scale_Value_user / (scale_box1[3][0] - scale_box1[0][0])) / 1000  # 微米转化为毫米
                   print(f"比例尺文字提取失败！采用用户输入的默认比例尺{Scale_Value_user}：1像素：{pixel_to_mm_ratio}mm")
                else:
                    pixel_to_mm_ratio = (500 / (scale_box2[3][0] - scale_box2[0][0])) / 1000
                    print(f'数字识别失败，比例尺尺度提取成功，采用默认比例尺数字（1：500） 1像素：{pixel_to_mm_ratio}mm')
            else:
                scale_value = abs(scale_value)
                if scale_box2 is None:
                    pixel_to_mm_ratio = (scale_value / (scale_box1[3][0] - scale_box1[0][0])) / 1000  # 微米转化为毫米
                    print(f'使用文字区域大小比例尺  1像素：{pixel_to_mm_ratio}mm')
                else:
                    # print(scale_box2)
                    pixel_to_mm_ratio = (scale_value / (scale_box2[3][0] - scale_box2[0][0])) / 1000  # 微米转化为毫米

                    print(f'成功提取比例尺  1像素：{pixel_to_mm_ratio}mm')
        # else:
            # print(f"比例尺矩形框提取失败！采用指定比例尺  1像素：{pixel_to_mm_ratio}mm")

        return  pixel_to_mm_ratio

# path = '../../KELI/all_rocks/data/images/1-5-3-5.jpg'

# p = Pixel_to_ratio(path,pixel_to_mm_ratio = 0.0001,min_area = 500,Scale_Value_user=500)

