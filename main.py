from face_detection import face_detection
from Delaunay import delaunay, delaunay2
import cv2
import numpy as np


# 通过返回的三角点建立索引
def index_find(point, landmarks):
    for index in range(len(landmarks)):
        if point == landmarks[index]:
            return index


# 根据原本三角形、结果三角形和差值三角形位置计算像素变换结果
def applyAffineTransform(src, srcTri, dstTri, size):
    # 获得仿射变换矩阵
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # 应用变换矩阵获得变换后的图像 INTER_LINEAR（线性插值）BORDER_REFLECT_101（边界最外层像素不会重复[ the outter-most pixels
    # (a or h) are not repeated]，后续融合效果好）
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # 获得最小外接矩形
    # 返回四个值，分别是x，y，w，h
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # 以外接矩形左上角为原点重设坐标
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))

    # 创建一个mask
    mask = np.zeros((r[3], r[2], 3), dtype='float32')
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # 获得原本三角形外接矩形区域的像素值
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # 图片权重合成
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # 通过mask将矩形转化为三角形，将计算结果放入图像数组
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


if __name__ == '__main__':
    # ted_cruz/hillary_clinton/Arnie/Bush/donald_trump
    # paths = "paths.txt"
    # fp = open(paths, 'r')
    # images = []

    # for each in fp.readlines():
    #     images.append(each.rstrip('\r\n'))
    ori_img_path = "images/ted_cruz.jpg"
    fin_img_path = "images/hillary_clinton.jpg"

    show_Tri = 0  # 是否显示三角
    frames = 40  # 帧数

    ori_img = cv2.imread(ori_img_path)
    fin_img = cv2.imread(fin_img_path)
    if ori_img is None or fin_img is None:
        print("Read img fail!")

    # 初始图片和结果图片人脸检测，获得特征点数组
    ori_landmarks = face_detection(ori_img)
    fin_landmarks = face_detection(fin_img)

    # 获得初始图片三角化的三角形点位
    ori_delaunay = delaunay(ori_img, ori_landmarks, 1)

    # 将获得的三角形点位转换为索引
    # 后续变化该索引不变
    tri_index = []
    for t in ori_delaunay:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        add = [(index_find(pt1, ori_landmarks), index_find(pt2, ori_landmarks), index_find(pt3, ori_landmarks))]
        tri_index.extend(add)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 20
    videoWriter = cv2.VideoWriter("result/video.avi", fourcc, fps, (600, 800))

    # 逐帧计算
    for k in range(0, frames + 1):
        alpha = k / frames
        landmarks_Middle = []
        # 人脸特征点插值
        for i in range(len(ori_landmarks)):
            x = int((1 - alpha) * ori_landmarks[i][0] + alpha * fin_landmarks[i][0])
            y = int((1 - alpha) * ori_landmarks[i][1] + alpha * fin_landmarks[i][1])
            landmarks_Middle.append((x, y))

        # 放中间结果
        imgMorph = np.zeros(ori_img.shape, dtype=ori_img.dtype)

        # 逐个三角形计算扭曲
        for j in range(len(tri_index)):
            # 获得点位索引
            x = tri_index[j][0]
            y = tri_index[j][1]
            z = tri_index[j][2]

            # 根据索引获得点坐标
            t1 = [ori_landmarks[x], ori_landmarks[y], ori_landmarks[z]]
            t2 = [fin_landmarks[x], fin_landmarks[y], fin_landmarks[z]]
            t = [landmarks_Middle[x], landmarks_Middle[y], landmarks_Middle[z]]

            # 对一个三角形做morphing
            morphTriangle(ori_img, fin_img, imgMorph, t1, t2, t, alpha)
        # 结果
        if show_Tri == 1:
            imgMorph_delaunay = delaunay2(imgMorph, tri_index, landmarks_Middle, (255, 255, 255))
            cv2.imshow("Morphed Face", np.uint8(imgMorph_delaunay))
            cv2.imwrite("result/frames/"+str(k)+".jpg",imgMorph_delaunay)
            videoWriter.write(imgMorph_delaunay)
            cv2.waitKey(50)
        else:
            cv2.imshow("Morphed Face", np.uint8(imgMorph))
            cv2.imwrite("result/frames/" + str(k) + ".jpg", imgMorph)
            videoWriter.write(imgMorph)
            cv2.waitKey(50)

    videoWriter.release()
