import cv2

def draw_text_box(frame, text, position, color=(255, 255, 255), font_scale=0.7, thickness=1, padding=5):
    """Draw text on translucent background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    w, h = text_size
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - h - padding), (x + w + 2 * padding, y + padding), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x + padding, y), font, font_scale, color, thickness, cv2.LINE_AA)
