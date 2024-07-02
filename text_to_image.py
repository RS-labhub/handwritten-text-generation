from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, font_path='QEBEV.ttf', font_size=30):
    font = ImageFont.truetype(font_path, font_size)
    lines = text.split('\n')
    max_width = max([font.getsize(line)[0] for line in lines])
    total_height = sum([font.getsize(line)[1] for line in lines])

    image = Image.new('RGB', (max_width, total_height), 'white')
    draw = ImageDraw.Draw(image)

    y_text = 0
    for line in lines:
        draw.text((0, y_text), line, font=font, fill='black')
        y_text += font.getsize(line)[1]

    image.show()

# Example usage
text = "This is an example of handwritten text generated by the model."
text_to_image(text, font_path='QEBEV.ttf')