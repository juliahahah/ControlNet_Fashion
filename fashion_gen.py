import cv2
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
import os

# 設定輸出資料夾
# 取得目前腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(script_dir, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 準備輸入圖片 (這裡我們下載一張範例圖片，你也可以換成自己的圖片路徑)
# 這裡使用一張全身照作為範例
IMAGE_URL = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
print(f"正在下載或讀取圖片: {IMAGE_URL}...")
init_image = load_image(IMAGE_URL)
init_image.save(os.path.join(OUTPUT_DIR, "original.png"))

# 2. 載入 OpenPose 預處理模型 (用於提取骨架/姿勢)
print("正在載入 OpenPose 偵測器...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# 提取姿勢
print("正在提取姿勢...")
pose_image = openpose(init_image)
pose_image.save(os.path.join(OUTPUT_DIR, "pose.png"))

# 3. 載入 ControlNet 和 Stable Diffusion 模型
print("正在載入 ControlNet 和 Stable Diffusion 模型 (這可能需要一點時間)...")
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", 
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# 使用更快的排程器 (Scheduler)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# 啟用 CPU Offload 以節省記憶體 (適合顯卡記憶體較小的環境)
pipe.enable_model_cpu_offload()

# 如果有安裝 xformers，啟用它以加速
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print("未啟用 xformers (可選):", e)

# 4. 定義時尚生成的 Prompt (提示詞)
# 這裡我們可以定義不同的服裝風格
fashion_prompts = [
    "a fashion model wearing a red silk evening gown, runway photography, high fashion, 8k, highly detailed",
    "a fashion model wearing a futuristic silver cyberpunk jacket and black pants, neon lights, urban street style, detailed",
    "a fashion model wearing a vintage floral summer dress, sunlight, nature background, soft lighting"
]

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, missing limbs, extra limbs"

# 5. 生成圖片
print("開始生成時尚圖片...")
generator = torch.Generator(device="cpu").manual_seed(42) # 固定隨機種子以重現結果

for i, prompt in enumerate(fashion_prompts):
    print(f"正在生成第 {i+1} 張: {prompt}")
    output = pipe(
        prompt,
        pose_image,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=20,
    )
    
    generated_image = output.images[0]
    save_path = os.path.join(OUTPUT_DIR, f"fashion_output_{i+1}.png")
    generated_image.save(save_path)
    print(f"已儲存: {save_path}")

print("所有圖片生成完畢！請查看 outputs 資料夾。")
