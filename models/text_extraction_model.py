#ocr

# from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
# model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cpu', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
# model = model.eval()
# def ocr(image_file):
# # input your test image
#     # image_file = 'C:\Users\vishal sharma\Desktop\project_root\data\input_images\ss.png'

# # plain texts OCR
#     res = model.chat(tokenizer, image_file, ocr_type='ocr')

# # format texts OCR:
# # res = model.chat(tokenizer, image_file, ocr_type='format')

# # fine-grained OCR:
# # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# # multi-crop OCR:
# # res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# # res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# # render the formatted OCR results:
# # res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

    
#     return res

import easyocr

def extract_text(image_path):
    """
    Extracts text from an image using EasyOCR.

    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Extracted text from the image.
    """
    extracted_text = ''
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader for English
    text = reader.readtext(image_path)
    extracted_text = '\n'.join([item[1] for item in text])
    return extracted_text

























