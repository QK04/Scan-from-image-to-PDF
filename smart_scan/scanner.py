import cv2
import numpy as np

def apply_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Làm nét chuyên sâu
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
    # Cân bằng sáng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Tẩy nền giấy
    _, final = cv2.threshold(gray, 220, 255, cv2.THRESH_TRUNC)
    final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
    return final

def resize_to_a4(img):
    # Tỉ lệ A4 chuẩn là 1 : 1.414
    target_ratio = 1.414
    h, w = img.shape[:2]
    current_ratio = h / w

    if current_ratio > target_ratio:
        # Ảnh quá dài -> Thêm trắng vào 2 bên
        new_w = int(h / target_ratio)
        new_h = h
    else:
        # Ảnh quá rộng -> Thêm trắng vào trên dưới
        new_h = int(w * target_ratio)
        new_w = w

    # Tạo khung trắng
    container = np.full((new_h, new_w), 255, dtype=np.uint8)
    # Chèn ảnh vào giữa khung
    x_offset = (new_w - w) // 2
    y_offset = (new_h - h) // 2
    container[y_offset:y_offset+h, x_offset:x_offset+w] = img
    
    # Resize về kích thước A4 tiêu chuẩn (pixel) để giảm nhẹ dung lượng PDF
    return cv2.resize(container, (1240, 1754), interpolation=cv2.INTER_AREA)

def apply_warp(img, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    filtered = apply_filter(warped)
    return resize_to_a4(filtered)

# Các hàm scan_document và manual_warp giữ nguyên logic gọi apply_warp mới
def scan_document(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            res = apply_warp(orig, approx.reshape(4, 2))
            _, buf = cv2.imencode('.jpg', res); return buf.tobytes()
    res = resize_to_a4(apply_filter(orig))
    _, buf = cv2.imencode('.jpg', res); return buf.tobytes()

def manual_warp(image_bytes, points):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pts = np.array([[p['x'], p['y']] for p in points], dtype="float32")
    res = apply_warp(img, pts)
    _, buf = cv2.imencode('.jpg', res); return buf.tobytes()