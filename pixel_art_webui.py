import gradio as gr
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os
from datetime import datetime

def reduce_colors(img, n_colors):
    img_np = np.array(img.convert("RGB"))
    h, w, _ = img_np.shape
    flat = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(flat)
    labels = kmeans.predict(flat)
    new_flat = kmeans.cluster_centers_[labels].astype("uint8")
    return Image.fromarray(new_flat.reshape(h, w, 3))

def pixel_art_convert_and_save(input_img, color_count, output_dir, output_width, output_height):
    if input_img is None:
        return None, "画像をアップロードしてください"
    if output_width < 1 or output_height < 1 or color_count < 2:
        return None, "パラメータを正しく設定してください"
    if not output_dir:
        return None, "出力フォルダパスを入力してください"
    if not os.path.isdir(output_dir):
        return None, f"指定されたフォルダが存在しません: {output_dir}"

    img = input_img.convert("RGB")

    # 指定ピクセル数で縮小（ピクセルアート用サイズ）
    small = img.resize((int(output_width), int(output_height)), resample=Image.NEAREST)

    # 色数減らし
    reduced = reduce_colors(small, color_count)

    # 出力画像は拡大しない（ピクセルアートサイズそのまま）
    pixel_art_img = reduced

    # 自動保存処理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pixelart_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    pixel_art_img.save(save_path)

    status_msg = f"変換完了・保存しました: {save_path}"

    return pixel_art_img, status_msg

# Gradio UI構築
with gr.Blocks() as demo:
    gr.Markdown("# Pixel Art Converter (Web UI)")

    with gr.Row():
        input_img = gr.Image(type="pil", label="アップロード画像", height=300)  
        output_img = gr.Image(type="pil", label="ピクセルアート出力", interactive=False, height=300)

    color_count = gr.Slider(2, 64, value=64, step=1, label="Color Count")
    output_width = gr.Number(value=32, label="出力ピクセル幅", precision=0)
    output_height = gr.Number(value=32, label="出力ピクセル高さ", precision=0)
    output_dir = gr.Textbox(value="./output", label="出力フォルダパス")

    status = gr.Textbox(label="状態", interactive=False)

    btn = gr.Button("変換＆自動保存")
    btn.click(fn=pixel_art_convert_and_save,
              inputs=[input_img, color_count, output_dir, output_width, output_height],
              outputs=[output_img, status])

demo.launch(inbrowser=True)
