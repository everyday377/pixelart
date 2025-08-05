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

def pixel_art_convert_and_save(input_img, pixel_size, color_count, scale, transparent, output_dir):
    if input_img is None:
        return None, "画像をアップロードしてください"
    if pixel_size < 1 or color_count < 2 or scale <= 0:
        return None, "パラメータを正しく設定してください"
    if not output_dir:
        return None, "出力フォルダパスを入力してください"
    if not os.path.isdir(output_dir):
        return None, f"指定されたフォルダが存在しません: {output_dir}"

    img = input_img.convert("RGBA")

    # 縮小（ピクセル化）
    small_w = max(1, int(img.width / pixel_size))
    small_h = max(1, int(img.height / pixel_size))
    small = img.resize((small_w, small_h), resample=Image.NEAREST)

    # 色数減らし
    reduced = reduce_colors(small, color_count)

    # 透過処理
    if transparent:
        reduced = reduced.convert("RGBA")
        alpha_small = img.getchannel("A").resize(reduced.size, Image.NEAREST)
        data = reduced.getdata()
        new_data = []
        for i, pix in enumerate(data):
            x = i % reduced.width
            y = i // reduced.width
            if alpha_small.getpixel((x, y)) == 0:
                new_data.append((0, 0, 0, 0))
            else:
                if len(pix) == 3:
                    new_data.append(pix + (255,))
                else:
                    new_data.append(pix)
        reduced.putdata(new_data)
    else:
        reduced = reduced.convert("RGB")

    # 拡大サイズ（倍率適用）
    out_w = max(1, int(img.width * scale))
    out_h = max(1, int(img.height * scale))

    pixel_art_img = reduced.resize((out_w, out_h), resample=Image.NEAREST)

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
        input_img = gr.Image(type="pil", label="アップロード画像",height=300)  
        output_img = gr.Image(type="pil", label="ピクセルアート出力", interactive=False,height=300)

    pixel_size = gr.Slider(2, 64, value=16, step=1, label="Pixel Size")
    color_count = gr.Slider(2, 64, value=64, step=1, label="Color Count")
    scale = gr.Number(value=1.0, label="倍率 (0より大きい実数)", precision=2)
    transparent = gr.Checkbox(value=False, label="背景透過PNGで保存")

    output_dir = gr.Textbox(value="./output", label="出力フォルダパス")

    status = gr.Textbox(label="状態", interactive=False)

    btn = gr.Button("変換＆自動保存")
    btn.click(fn=pixel_art_convert_and_save,
              inputs=[input_img, pixel_size, color_count, scale, transparent, output_dir],
              outputs=[output_img, status])

demo.launch(inbrowser=True)
