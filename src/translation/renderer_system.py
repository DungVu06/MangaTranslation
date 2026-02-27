import textwrap

from PIL import Image, ImageDraw, ImageFont

class MangaRenderer:
    def __init__(self):
        pass

    def _draw_text_centered(self, draw, text, box, max_font_size=40):
        xmin, ymin, xmax, ymax = box
        box_width = xmax - xmin
        box_height = ymax - ymin

        font_size = max_font_size
        
        while font_size > 8:
            try:
                font = ImageFont.load_default(size=font_size)
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

    def render_translated_image(self, image_path, translated_data, output_path="final_translated_manga.jpg"):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for item in translated_data:
            text_en = item.get("english_text", "")
            if not text_en:
                continue

            xmin, ymin, xmax, ymax = map(int, item["coordinates"])

            draw.rectangle([xmin, ymin, xmax, ymax], fill="white")
            self._draw_text_centered(draw, text_en, [xmin, ymin, xmax, ymax])

        img.save(output_path, quality=95)
        
        # img.show()