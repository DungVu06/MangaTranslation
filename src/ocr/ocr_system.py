import numpy as np

from manga_ocr import MangaOcr
from PIL import Image

class MangaTextExtractor:
    def __init__(self):
        self.mocr = MangaOcr()

    def _calculate_ioa(self, text_box, frame_box):
        x_left = max(text_box[0], frame_box[0])
        y_top = max(text_box[1], frame_box[1])
        x_right = min(text_box[2], frame_box[2])
        y_bottom = min(text_box[3], frame_box[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        text_box_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
        
        return intersection_area / float(text_box_area)
    
    def sort_reading_order(self, text_boxes, frame_boxes):
        # ====================================================
        # BƯỚC 1: XẾP KHUNG TRUYỆN (FRAME)
        # ====================================================
        # 1.1 Sắp xếp tạm các khung theo chiều dọc (Từ Trên xuống Dưới)
        frames_sorted_by_y = sorted(frame_boxes, key=lambda b: b[1])
        
        rows = []
        for frame in frames_sorted_by_y:
            added = False
            for row in rows:
                # Lấy khung đầu tiên của hàng làm chuẩn để so sánh
                base_frame = row[0]
                
                # Tính độ giao nhau theo trục Y
                overlap_y = max(0, min(base_frame[3], frame[3]) - max(base_frame[1], frame[1]))
                frame_height = frame[3] - frame[1]
                
                # Nếu khung này giao nhau ít nhất 30% chiều cao với hàng hiện tại -> Cho vào chung 1 hàng
                if overlap_y > 0.3 * frame_height:
                    row.append(frame)
                    added = True
                    break
            
            # Nếu không thuộc hàng nào (nằm tít dưới), tạo hàng mới
            if not added:
                rows.append([frame])
                
        # 1.2 Xếp các khung TRONG TỪNG HÀNG (Từ Phải qua Trái)
        sorted_frames = []
        for row in rows:
            # Dùng xmax (-b[2]) để canh mép phải chuẩn hơn
            row_sorted_x = sorted(row, key=lambda b: -b[2]) 
            sorted_frames.extend(row_sorted_x)
            
        # ====================================================
        # BƯỚC 2: GOM CHỮ VÀO KHUNG (Dùng IoA)
        # ====================================================
        clusters = {i: [] for i in range(len(sorted_frames))}
        unassigned_texts = []

        for text in text_boxes:
            best_frame_idx = -1
            max_ioa = 0.0

            for i, frame in enumerate(sorted_frames):
                ioa = self._calculate_ioa(text, frame)
                if ioa > max_ioa:
                    max_ioa = ioa
                    best_frame_idx = i

            if max_ioa > 0.2:
                clusters[best_frame_idx].append(text)
            else:
                unassigned_texts.append(text)

        # ====================================================
        # BƯỚC 3: XẾP CHỮ TRONG KHUNG (Từ Phải qua Trái)
        # ====================================================
        final_ordered_texts = []
        for i in range(len(sorted_frames)):
            # Bên trong một khung, ta cũng ưu tiên Phải -> Trái, Trên -> Dưới
            cluster_texts = sorted(clusters[i], key=lambda b: (-b[2], b[1]))
            final_ordered_texts.extend(cluster_texts)
            
        unassigned_texts = sorted(unassigned_texts, key=lambda b: (-b[2], b[1]))
        final_ordered_texts.extend(unassigned_texts)

        return final_ordered_texts

    def extract_text(self, img_path, text_boxes, frame_boxes):
        img = Image.open(img_path).convert("RGB")
        ordered_boxes = self.sort_reading_order(text_boxes, frame_boxes)
        
        extracted_texts = []

        for i, box in enumerate(ordered_boxes):
            xmin, ymin, xmax, ymax = map(int,box)
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            text = self.mocr(cropped_img)

            extracted_texts.append({
                "box_id": i,
                "coordinates": [xmin, ymin, xmax, ymax],
                "japanese_text": text
            })
        return extracted_texts
    
if __name__ == "__main__":
    img = "./data/inference_data/doraemon_2.jpg"
    dummy_boxes = [
        [155, 99, 192, 152]
    ]
    extractor = MangaTextExtractor()
    results = extractor.extract_text(img, dummy_boxes, dummy_boxes)
    print(results)