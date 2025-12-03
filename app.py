import gradio as gr
import cv2
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import os

# 全域變數儲存模型，避免重複載入
openpose = None
pipe = None

def load_models():
    global openpose, pipe
    if openpose is None:
        print("正在載入 OpenPose 偵測器...")
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    if pipe is None:
        print("正在載入 ControlNet 和 Stable Diffusion Inpainting 模型...")
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose", 
            torch_dtype=torch.float16
        )

        # 使用 Inpainting Pipeline 來保留原圖的未遮罩區域 (如臉部、背景)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("未啟用 xformers (可選):", e)
    
    return "模型載入完成！"

def generate_fashion(input_dict, prompt, negative_prompt, num_steps, seed):
    if input_dict is None or input_dict["background"] is None:
        return None, None
    
    image = input_dict["background"]
    
    # 處理遮罩：從 ImageEditor 的圖層中提取
    if not input_dict["layers"]:
        # 如果沒有圖層，嘗試看看是否有 composite (有些版本行為不同)
        # 但通常 layers 會有塗抹內容
        return None, None
        
    # 合併所有圖層的 Alpha 通道作為遮罩
    mask = Image.new("L", image.size, 0)
    for layer in input_dict["layers"]:
        # layer 是 RGBA，取出 Alpha 通道
        layer_alpha = layer.split()[-1]
        mask = Image.fromarray(np.maximum(np.array(mask), np.array(layer_alpha)))

    if image is None:
        return None, None
    
    if pipe is None or openpose is None:
        load_models()

    # 1. 提取姿勢
    print("正在提取姿勢...")
    # 確保圖片是 PIL Image 格式
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 調整大小以符合模型需求 (建議 512x512 或其倍數)
    w, h = image.size
    # 簡單縮放至 512x512 進行處理
    process_image = image.resize((512, 512)).convert("RGB")
    process_mask = mask.resize((512, 512)).convert("RGB") # 轉換為 RGB 以避免某些版本的相容性問題
        
    pose_image = openpose(process_image)

    # 2. 生成圖片
    print(f"正在生成圖片: {prompt}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    
    output = pipe(
        prompt,
        image=process_image,
        mask_image=process_mask,
        control_image=pose_image,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=int(num_steps),
        height=512,
        width=512,
    )
    
    generated_image = output.images[0]
    
    # 將生成的圖片 resize 回原始大小 (可選)
    generated_image = generated_image.resize((w, h))
    pose_image = pose_image.resize((w, h))
    
    return pose_image, generated_image

# 定義 Gradio 介面
with gr.Blocks(title="AI 時尚設計師") as demo:
    gr.Markdown("# 👗 AI 時尚設計師 (虛擬試穿版)")
    gr.Markdown("上傳人物照片，並**使用畫筆塗抹想要更換的衣服區域**，AI 將為你生成全新的時尚穿搭，同時保留模特兒的臉部和身體特徵！")
    
    with gr.Row():
        with gr.Column():
            # 使用 ImageEditor 讓使用者可以塗抹遮罩
            input_img = gr.ImageEditor(label="上傳照片並塗抹衣服區域", type="pil")
            prompt_text = gr.Textbox(
                label="服裝描述 (Prompt)", 
                placeholder="例如: a fashion model wearing a red silk evening gown, runway photography...",
                value="a fashion model wearing a red silk evening gown, runway photography, high fashion, 8k, highly detailed"
            )
            neg_prompt_text = gr.Textbox(
                label="負面描述 (Negative Prompt)", 
                value="monochrome, lowres, bad anatomy, worst quality, low quality, missing limbs, extra limbs"
            )
            with gr.Accordion("進階設定", open=False):
                steps_slider = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="推論步數 (Steps)")
                seed_number = gr.Number(value=42, label="隨機種子 (Seed)")
            
            run_btn = gr.Button("開始生成", variant="primary")
        
        with gr.Column():
            with gr.Row():
                pose_output = gr.Image(label="偵測到的骨架 (Pose)")
                final_output = gr.Image(label="生成結果")

    # 綁定事件
    run_btn.click(
        fn=generate_fashion,
        inputs=[input_img, prompt_text, neg_prompt_text, steps_slider, seed_number],
        outputs=[pose_output, final_output]
    )

    # 啟動時預先載入模型 (可選，若不想啟動時卡住可註解掉)
    # load_models()

if __name__ == "__main__":
    demo.launch()
