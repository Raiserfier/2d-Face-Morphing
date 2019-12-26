import dlib
import cv2


def face_detection(img):
    # Dlib 检测器和预测器
    global landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/dlib/shape_predictor_81_face_landmarks.dat')

    # 读取图像文件
    h, w = img.shape[:2]
    img_rd = img
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数
    faces = detector(img_gray, 0)

    # 标 81 个点
    if len(faces) != 0:
        # 检测到人脸
        landmarks = []
        for i in range(len(faces)):
            # 取特征点坐标
            for p in predictor(img_rd, faces[i]).parts():
                landmarks.append((p.x, p.y))
    else:
        print("No face!")

    # 加入边缘点
    add = [(0, 0), (0, int(h / 2)), (0, int(h - 1)), (int(w / 2), h - 1),
           (w - 1, h - 1), (w - 1, int(h / 2)), (w - 1, 0), (int(w / 2), 0)]
    landmarks.extend(add)

    return landmarks
