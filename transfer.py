from PIL import Image
import os
def read_path(file_pathname):
    # for filename in os.listdir(file_pathname):
    #     pic = Image.open(r''+file_pathname+'\\'+filename)
    #     pic = pic.convert('RGBA')  # 转为RGBA模式
    #     width, height = pic.size
    #     array = pic.load()  # 获取图片像素操作入口
    #     for i in range(width):
    #         for j in range(height):
    #             pos = array[i, j]  # 获得某个像素点，格式为(R,G,B,A)元组
    #             #黑色背景，白色背景改255或者>=240
    #             isEdit = (sum([1 for x in pos[0:3] if x == 0]) == 3)
    #             if isEdit:
    #                 # 更改为透明
    #                 array[i, j] = (255, 255, 255, 0)

    #     # 保存图片
        pic = Image.open(r''+file_pathname)
        pic = pic.convert('RGBA')  # 转为RGBA模式
        width, height = pic.size
        array = pic.load()  # 获取图片像素操作入口
        for i in range(width):
            for j in range(height):
                pos = array[i, j]  # 获得某个像素点，格式为(R,G,B,A)元组
                #黑色背景，白色背景改255或者>=240
                isEdit = (sum([1 for x in pos[0:3] if x > 200]) == 3)
                if isEdit:
                    # 更改为透明
                    array[i, j] = (255, 255, 255, 0)
        pic.save(file_pathname)

read_path("figures/exp.png")



