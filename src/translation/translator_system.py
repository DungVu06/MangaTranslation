from deep_translator import GoogleTranslator
import time

class MangaTranslator:
    def __init__(self, src_lang="ja", target_lang="en"):
        self.translator = GoogleTranslator(src_lang, target_lang)

    def translate_with_context(self, ocr_results):
        if not ocr_results:
            return []
        full_jp_text = "\n".join([res["japanese_text"] for res in ocr_results])

        full_en_text = self.translator.translate(full_jp_text)
        translated_sentences = full_en_text.split("\n")

        translated_results = []
        for i, result in enumerate(ocr_results):
            new_result = result.copy()
            new_result["english_text"] = translated_sentences[i].strip()
            translated_results.append(new_result)

        return translated_results
    
if __name__ == "__main__":
    dummy_ocr_results = [
        {"box_id": 1, "coordinates": [10, 10, 50, 50], "japanese_text": "誰がケーキを盗んだの？"}, # Who stole the cake?
        {"box_id": 2, "coordinates": [60, 60, 90, 90], "japanese_text": "あいつだ！"}            # It's him!
    ]

    translator = MangaTranslator()
    results = translator.translate_with_context(dummy_ocr_results)
    print(results)