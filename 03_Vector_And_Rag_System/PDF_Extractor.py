import os

from io import BytesIO
from PIL import Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage, LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    """从PDF文件中（按指定页码）提取文字"""
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'

    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) > min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''

    if buffer:
        paragraphs.append(buffer)
    return paragraphs

def extract_images_from_pdf(pdf_path, output_dir="extracted_images"):
    """
    从PDF文件中提取图片

    参数:
    pdf_path: PDF文件路径
    output_dir: 图片输出目录
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_count = 0

    try:
        # 遍历PDF的每一页
        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            print(f"正在处理第 {page_num + 1} 页...")

            # 递归查找页面的所有元素
            image_count = find_images_in_layout(page_layout, output_dir, page_num, image_count)

    except Exception as e:
        print(f"处理PDF时出错: {e}")

    print(f"总共提取了 {image_count} 张图片")

def find_images_in_layout(layout, output_dir, page_num, image_count):
    """
    递归查找布局中的图片元素
    """
    for element in layout:
        if isinstance(element, LTFigure):
            # 如果是图形对象，递归查找其中的图片
            image_count = find_images_in_layout(element, output_dir, page_num, image_count)
        elif isinstance(element, LTImage):
            # 找到图片，保存它
            try:
                # 获取图片原始数据
                image_data = element.stream.get_rawdata()

                # 使用PIL打开图片
                image = Image.open(BytesIO(image_data))

                # 生成文件名并保存
                image_filename = f"page_{page_num}_image_{image_count+1}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)

                print(f"已保存图片: {image_filename} (尺寸: {image.size})")
                image_count += 1

            except Exception as e:
                print(f"处理图片时出错: {e}")
                # 如果PIL无法处理，尝试保存原始数据
                try:
                    image_filename = f"page_{page_num}_image_{image_count+1}.bin"
                    image_path = os.path.join(output_dir, image_filename)

                    with open(image_path, "wb") as img_file:
                        img_file.write(image_data)

                    print(f"已保存原始数据: {image_filename}")
                    image_count += 1
                except:
                    print("无法保存图片数据")

    return image_count



extracted_paragraphs = extract_text_from_pdf("./docs/llama2.pdf", min_line_length=10)
for para in extracted_paragraphs[:10]:
    print(para + '\n')