# %%
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pathlib
import os
import tempfile

# 설정된 경로들
parent_dir = str(pathlib.Path(os.getcwd()).parents[0])

dataset = 'sentiment-0'
dropout = 'quantile'
dropout_rate = 0.8
data_dir = 'results/{}'.format(dataset)
rl_model  = 'xglm'
# rl_model = 'xglm_large'
# rl_model = 'opt_large'
folder_dir = 'ref=opt_large_ref_dec=stochastic_n_epoch=5_lr=5e-06_dropout={}_dropout_rate={}'.format(dropout, dropout_rate)
file_dir = os.path.join(parent_dir, data_dir, rl_model, folder_dir)

# 대상 PDF 파일들
pdf_files = [
    'dropout_heatmap_0_epoch.pdf',
    'dropout_heatmap_1_epoch.pdf',
    'dropout_heatmap_2_epoch.pdf',
    'dropout_heatmap_3_epoch.pdf',
    'dropout_heatmap_4_epoch.pdf'
]


# 트리밍된 이미지를 저장할 임시 리스트
trimmed_images = []


# 임시 폴더 설정
output_pdf_path = os.path.join(file_dir, 'combined_trimmed_resized.pdf')
temp_dir = tempfile.TemporaryDirectory()

def trim_image(image):
    # 이미지의 알파 채널이 있으면 제거하고 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # 그레이스케일로 변환
    gray_image = image.convert('L')
    # 테두리의 배경색을 기준으로 트리밍
    bg_color = gray_image.getpixel((0, 0))
    bbox = gray_image.point(lambda p: p != bg_color).getbbox()
    return image.crop(bbox)

def process_and_add_text(image, text):
    fig, ax = plt.subplots(figsize=(image.width / 100.0, image.height / 100.0), dpi=100)
    ax.imshow(image, aspect='auto')
    ax.set_axis_off()  # 모든 축을 제거합니다.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # 텍스트 크기를 이미지 높이에 맞추어 추가
    plt.text(-0.03, 0.5, text, fontsize=image.height // 2, ha='right', va='center', transform=ax.transAxes, color='black')
    return fig

all_images = []

for idx, pdf_file in enumerate(pdf_files):
    pdf_path = os.path.join(file_dir, pdf_file)
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 이미지의 좌우 공백 제거
        trimmed_img = trim_image(img)

        # 이미지 크기 조정
        new_width = int(trimmed_img.width * 1.0)
        new_height = trimmed_img.height // 1
        resized_img = trimmed_img.resize((new_width, new_height))
        
        # 텍스트 추가
        fig = process_and_add_text(resized_img, f'epoch={idx+1}')
        
        # 임시 파일에 이미지 저장
        temp_img_path = os.path.join(temp_dir.name, f"temp_img_{idx}.png")
        fig.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 임시 파일로부터 이미지를 불러옴
        temp_img = Image.open(temp_img_path)
        all_images.append(temp_img)
        
    doc.close()

# 모든 이미지를 세로로 쌓기
total_height = sum(img.height for img in all_images)
max_width = max(img.width for img in all_images)

combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
y_offset = 0
for img in all_images:
    combined_img.paste(img, (0, y_offset))
    y_offset += img.height

# 큰 이미지를 PDF로 저장
combined_img_path = os.path.join(temp_dir.name, "combined_img.png")
combined_img.save(combined_img_path)

# PDF 생성
c = canvas.Canvas(output_pdf_path, pagesize=letter)
img_width, img_height = combined_img.size
c.setPageSize((img_width, img_height))
c.drawImage(combined_img_path, 0, 0, width=img_width, height=img_height)
c.showPage()
c.save()

temp_dir.cleanup()
# %%
