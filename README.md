# AI Fashion Designer (Virtual Try-On Project)

這是一個基於生成式 AI 技術的虛擬試穿與時尚設計專案。透過結合 **Stable Diffusion**、**ControlNet** 與 **Inpainting** 技術，使用者可以上傳人物照片，並指定想要更換的服裝區域，AI 將在保留人物原始姿勢、臉部特徵與背景的前提下，生成全新的時尚穿搭。

## 核心技術架構 (Technical Architecture)

本專案整合了多種先進的電腦視覺與生成式模型技術：

### 1. Stable Diffusion Inpainting
*   **模型核心**: 使用 `runwayml/stable-diffusion-inpainting` 作為基礎模型。
*   **功能**: 不同於一般的文生圖 (Text-to-Image)，Inpainting 模型專注於「局部重繪」。它能理解遮罩 (Mask) 區域，並根據提示詞 (Prompt) 填補該區域，同時確保填補內容與周圍未遮罩區域（如背景、皮膚）的無縫融合。

### 2. ControlNet (OpenPose)
*   **模型**: `fusing/stable-diffusion-v1-5-controlnet-openpose`
*   **預處理器**: `lllyasviel/ControlNet` (OpenPoseDetector)
*   **作用**: 單純的 Inpainting 有時會導致肢體變形或姿勢改變。我們引入 ControlNet OpenPose 來提取人物的骨架姿勢，並將其作為額外的條件輸入給 Stable Diffusion。這確保了生成的服裝會精確地貼合人物原本的動作和體型。

### 3. Gradio UI
*   **介面**: 使用 Python 的 `gradio` 庫構建互動式網頁介面。
*   **功能**: 
    *   整合 `ImageEditor` 元件，支援使用者直接在瀏覽器中進行圖片塗抹 (Masking)。
    *   即時參數調整（提示詞、步數、種子碼）。
    *   並排顯示骨架偵測結果與最終生成圖。

## 處理流程 (Pipeline Workflow)

1.  **輸入處理 (Input Processing)**:
    *   接收使用者上傳的原始圖片與繪製的遮罩 (Mask)。
    *   將圖片與遮罩統一縮放至 512x512 解析度 (模型最佳輸入尺寸)。
    *   將遮罩轉換為 RGB 格式以符合 Tensor 運算需求。

2.  **姿勢提取 (Pose Extraction)**:
    *   使用 OpenPose 預處理器分析輸入圖片，偵測人體關鍵點 (Keypoints)。
    *   生成一張骨架圖 (Pose Map)，作為 ControlNet 的引導圖。

3.  **擴散生成 (Diffusion Generation)**:
    *   **Pipeline**: `StableDiffusionControlNetInpaintPipeline`
    *   **輸入**: 
        *   `prompt`: 服裝描述 (如 "red silk evening gown")。
        *   `image`: 原始圖片 (作為底圖)。
        *   `mask_image`: 使用者塗抹的區域 (告訴 AI 哪裡需要重繪)。
        *   `control_image`: 骨架圖 (限制生成的形狀與姿勢)。
    *   **推論**: 模型在 Latent Space 中進行去噪過程，逐步生成符合描述且貼合骨架的服裝紋理。

4.  **後處理 (Post-processing)**:
    *   將生成的 512x512 圖片縮放回原始圖片尺寸。
    *   輸出最終合成結果。

## 檔案結構 (File Structure)

*   `app.py`: 主要的應用程式入口。包含 Gradio 介面定義、模型載入邏輯與生成函式。
*   `fashion_gen.py`: (舊版) 批次生成腳本，不含 UI，主要用於測試基本的 ControlNet 生成功能。
*   `requirements.txt`: 專案依賴套件列表。
*   `outputs/`: 存放生成結果的資料夾。

## 環境需求 (Requirements)

*   Python 3.8+
*   CUDA 支援的 NVIDIA 顯示卡 (建議 VRAM 8GB 以上以獲得最佳效能)
    *   *註: 程式碼已包含 `enable_model_cpu_offload()` 與 `enable_xformers_memory_efficient_attention()` 優化，可在較低配置下運行，但速度會受影響。*

## 安裝與執行 (Installation & Usage)

1.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **啟動 Web UI**:
    ```bash
    python app.py
    ```

3.  **操作**:
    *   打開瀏覽器 (預設 `http://127.0.0.1:7860`)。
    *   上傳圖片 -> 使用畫筆塗抹衣服 -> 輸入提示詞 -> 點擊生成。
    *   https://youtu.be/oBC7mv9sKoY
