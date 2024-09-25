# image summarization---------------------------------------------------------------------------------------------------------


import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

url = '/content/download (1).jpg'
raw_image = Image.open(url).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))


#text summarizer --------------------------------------------------------------------------------------------------------------
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

y=summarizer(ocr_output, max_length=130, min_length=30, do_sample=False)


