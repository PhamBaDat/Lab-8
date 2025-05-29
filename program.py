import cv2
import numpy as np

def create_image_pyramid():
    img = cv2.imread('images/variant-9.png')
    if img is None:
        print("can't read photo variant-9.png")
        return
    
    pyramid = [img]
    for i in range(3):  
        img = cv2.pyrDown(img)
        pyramid.append(img)
    
    for i, level in enumerate(pyramid):
        cv2.imshow(f'Pyramid Level {i}', level)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_pattern_in_roi(roi_gray):
    h, w = roi_gray.shape
    half_h, half_w = h // 2, w // 2

    parts = [
        roi_gray[0:half_h, 0:half_w],       # phần trên trái
        roi_gray[0:half_h, half_w:w],       # phần trên phải
        roi_gray[half_h:h, 0:half_w],       # phần dưới trái
        roi_gray[half_h:h, half_w:w],       # phần dưới phải
    ]

    means = [np.mean(part) for part in parts]

    thresh = 100

    pattern = [
        means[0] < thresh,
        means[1] > thresh,
        means[2] > thresh,
        means[3] < thresh,
    ]

    return pattern == [True, True, True, True] or pattern == [True, False, False, True] or \
           pattern == [False, True, True, False] or pattern == [False, False, False, False]

def is_circular(roi_gray, threshold=0.7):
    _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > threshold  # Gần 1 là hình tròn chuẩn

def detect_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 150)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=50, minRadius=20, maxRadius=100)


    detected_markers = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if y-r < 0 or y+r > frame.shape[0] or x-r < 0 or x+r > frame.shape[1]:
                continue

            roi = gray[y-r:y+r, x-r:x+r]

            if is_circular(roi) and check_pattern_in_roi(roi):
                top_left = (x - r, y - r)
                bottom_right = (x + r, y + r)
                detected_markers.append((top_left, bottom_right))

    return detected_markers

def detect_marker_multi_scale(frame, scales=[1.0, 0.75, 0.5, 0.25]):
    all_markers = []
    h0, w0 = frame.shape[:2]

    for scale in scales:
        resized = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
        markers = detect_marker(resized)

        # Scale tọa độ marker về ảnh gốc
        for (top_left, bottom_right) in markers:
            tl = (int(top_left[0] / scale), int(top_left[1] / scale))
            br = (int(bottom_right[0] / scale), int(bottom_right[1] / scale))
            all_markers.append((tl, br))

    return all_markers

def track_label_with_fly():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera")
        return

    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    if fly is None:
        print("Can't read fly64.png")
        fly = None  

    centers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read frame from camera")
            break

        detected_markers = detect_marker_multi_scale(frame)
        for (top_left, bottom_right) in detected_markers:
            x_center = (top_left[0] + bottom_right[0]) // 2
            y_center = (top_left[1] + bottom_right[1]) // 2
            centers.append((x_center, y_center))

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

            if fly is not None:
                fly_h, fly_w = fly.shape[:2]
                top_left_x = x_center - fly_w // 2
                top_left_y = y_center - fly_h // 2

                if (top_left_x >= 0 and top_left_y >= 0 and
                    top_left_x + fly_w <= frame.shape[1] and
                    top_left_y + fly_h <= frame.shape[0]):
                    if fly.shape[2] == 4:  # alpha channel
                        alpha_s = fly[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        roi = frame[top_left_y:top_left_y+fly_h, top_left_x:top_left_x+fly_w]
                        for c in range(3):
                            roi[:, :, c] = (alpha_s * fly[:, :, c] + alpha_l * roi[:, :, c])
                    else:
                        frame[top_left_y:top_left_y+fly_h, top_left_x:top_left_x+fly_w] = fly

        cv2.imshow('Tracking Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if centers:
        avg_x = sum(c[0] for c in centers) / len(centers)
        avg_y = sum(c[1] for c in centers) / len(centers)
        print(f"Average coordinates during the session: ({avg_x:.2f}, {avg_y:.2f})")
    else:
        print("No label detected in session.")

if __name__ == "__main__":
    print("Part 1: Creating the pyramid image")
    create_image_pyramid()
    print("\nPart 2 + 3 & Additional Task: Trace the Label and Add the Fly")
    track_label_with_fly()