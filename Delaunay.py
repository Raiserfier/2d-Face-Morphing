import cv2


def delaunay(img, landmarks, return_type):
    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Create an array of points.
    points = []

    # Add points
    for item in landmarks:
        points.append(item)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Get delaunay triangles
    triangleList = subdiv.getTriangleList()

    if return_type == 0 or return_type == 2:
        # Draw delaunay triangles
        draw_delaunay(img_orig, triangleList, (255, 255, 255))

        # Draw points
        for p in points:
            draw_point(img_orig, p, (0, 0, 255))

        if return_type == 0:
            return img_orig, triangleList
        if return_type == 2:
            return img_orig

    elif return_type == 1:
        return triangleList


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)


# Draw delaunay triangles
def draw_delaunay(img, triangleList, delaunay_color):
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def delaunay2(img, triangleIndex, landmarks, delaunay_color):
    size = img.shape
    r = (0, 0, size[1], size[0])

    for j in range(len(triangleIndex)):
        x = triangleIndex[j][0]
        y = triangleIndex[j][1]
        z = triangleIndex[j][2]

        pt1 = landmarks[x]
        pt2 = landmarks[y]
        pt3 = landmarks[z]

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    return img
