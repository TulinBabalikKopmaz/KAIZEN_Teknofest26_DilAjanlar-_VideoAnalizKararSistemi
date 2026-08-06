
from fastapi import FastAPI, Request
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import uvicorn
import io
import base64

app = FastAPI()

print("Model optimize edilmiş ayarlarla VRAM'e yükleniyor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print("API Sunucusu Hazır!")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    max_tokens = data.get("max_tokens", 128)
    temperature = data.get("temperature", 0.2)
    
    # Gelen base64 görselin çözünürlüğünü VRAM patlamaması için optimize et (Max 768px)
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    img_data = item["image_url"]["url"]
                    if "base64," in img_data:
                        header, encoded = img_data.split("base64,", 1)
                        img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                        img.thumbnail((768, 768)) # Çözünürlüğü düşürerek OOM'yi engelle
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        new_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_data = f"data:image/jpeg;base64,{new_base64}"
                    
                    item["type"] = "image"
                    item["image"] = img_data
                    del item["image_url"]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            temperature=temperature,
            repetition_penalty=1.1
        )
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return {"choices": [{"message": {"content": output_text}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
