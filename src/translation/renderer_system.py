import textwrap
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

class MangaRenderer:
    def __init__(self, font_path="./fonts/ComicNeue-Bold.ttf"):
        self.font_path = font_path

    def _draw_text_centered(self, draw, text, box, max_font_size=40):
        xmin, ymin, xmax, ymax = box
        box_width = xmax - xmin 
        box_height = ymax - ymin

        font_size = max_font_size
        
        while font_size > 10:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except TypeError:
                font = ImageFont.load_default()

            avg_char_width = font.getlength("a") if hasattr(font, 'getlength') else font_size * 0.5
            max_chars_per_line = max(1, int((box_width * 0.9) / avg_char_width))

            lines = textwrap.wrap(text, width=max_chars_per_line)

            line_spacing = 4
            total_text_height = len(lines) * font_size + (len(lines) - 1) * line_spacing

            if total_text_height <= (box_height * 0.9):
                break
            
            font_size -= 2

        current_y = ymin + (box_height - total_text_height) / 2

        for line in lines:
            line_width = font.getlength(line) if hasattr(font, 'getlength') else len(line) * font_size * 0.5
            current_x = xmin + (box_width - line_width) / 2
            
            draw.text((current_x, current_y), line, fill="black", font=font)
            current_y += font_size + line_spacing

    def render_translated_image(self, image_path, translated_data, output_path):
        img_cv2 = cv2.imread(image_path)
        
        for item in translated_data:
            text_en = item.get("english_text", "")

            xmin, ymin, xmax, ymax = map(int, item["coordinates"])
           
            ymin, ymax = max(0, ymin), min(img_cv2.shape[0], ymax)
            xmin, xmax = max(0, xmin), min(img_cv2.shape[1], xmax)

            roi = img_cv2[ymin:ymax, xmin:xmax]
            if roi.size == 0: continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            H, W = roi.shape[:2]
            box_area = H * W
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                touches_edge = (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)
                
                if area > box_area * 0.2 and touches_edge:
                    cv2.drawContours(mask, [cnt], -1, 0, -1)

            kernel = np.ones((6, 6), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            inpainted_roi = cv2.inpaint(roi, mask, inpaintRadius=12, flags=cv2.INPAINT_NS)
            img_cv2[ymin:ymax, xmin:xmax] = inpainted_roi

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        for item in translated_data:
            text_en = item.get("english_text", "")
            # text_en = ""
            xmin, ymin, xmax, ymax = map(int, item["coordinates"])
            self._draw_text_centered(draw, text_en, [xmin, ymin, xmax, ymax])
        
        img_pil.save(output_path, quality=95)

if __name__ == "__main__":
    renderer = MangaRenderer()
    print("hello")