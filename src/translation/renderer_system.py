import textwrap
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

class MangaRenderer:
    def __init__(self, font_path="./fonts/ComicNeue-Bold.ttf"):
        self.font_path = font_path

    def _draw_text_centered(self, draw, text, box, max_font_size=40, min_font_size=10):
        xmin, ymin, xmax, ymax = box
        box_width = xmax - xmin 
        box_height = ymax - ymin

        if not text.strip():
            return
        
        font_cache = {}
        def get_font(size):
            if size not in font_cache:
                try:
                    font_cache[size] = ImageFont.truetype(self.font_path, size)
                except TypeError:
                    font_cache[size] = ImageFont.load_default()
            return font_cache[size]
        
        def wrap_text(text, font):
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = word if current_line == "" else current_line + " " + word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]

                if width <= box_width * 0.9:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)

            return lines

        low, high = min_font_size, max_font_size
        best_size = min_font_size
        best_lines = []

        while low <= high:
            mid = (low + high) // 2
            font = get_font(mid)
            lines = wrap_text(text, font)

            line_spacing = int(mid * 0.2)
            total_height = sum(
                draw.textbbox((0, 0), line, font=font)[3] - 
                draw.textbbox((0, 0), line, font=font)[1]
                for line in lines
            ) + (len(lines) - 1) * line_spacing

            if total_height <= box_height * 0.9:
                best_size = mid
                best_lines = lines
                low = mid + 1
            else:
                high = mid - 1
        
        font = get_font(best_size)
        line_spacing = int(best_size * 0.2)

        total_height = sum(
            draw.textbbox((0, 0), line, font=font)[3] - 
            draw.textbbox((0, 0), line, font=font)[1]
            for line in best_lines
        ) + (len(lines) - 1) * line_spacing

        current_y = ymin + (box_height - total_height) / 2

        for line in best_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            current_x = xmin + (box_width - line_width) / 2
            draw.text((current_x, current_y), line, fill="black", font=font)

            current_y += line_height + line_spacing

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