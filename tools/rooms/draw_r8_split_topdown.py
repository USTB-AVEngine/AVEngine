#!/usr/bin/env python3
"""Draw the temporary R8 split result; does not modify source data."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/data/smy/room_split_experiments/00006_20260907_v9/R8_split_topdown.png")

def font(size):
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def main():
    original = (-4.897, -5.236, 0.775, 4.707)
    sofa = (-4.5, -0.8, -1.0, 2.2)     # apartment_R8_01
    dining = (-4.5, -4.5, -1.0, -0.8) # apartment_R8_02
    camera_sofa = (-2.75, 0.70)
    camera_dining = (-2.397, -1.986)
    margin, scale = 0.65, 105
    x0, z0, x1, z1 = original
    lo_x, lo_z, hi_x, hi_z = x0-margin, z0-margin, x1+margin, z1+margin
    W, H = int((hi_x-lo_x)*scale)+180, int((hi_z-lo_z)*scale)+170
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    title, body, small = font(28), font(20), font(17)
    def p(x, z):
        return (int(80+(x-lo_x)*scale), int(80+(hi_z-z)*scale))
    def rect(box, outline, fill, width=5):
        a, b = p(box[0], box[3]), p(box[2], box[1])
        draw.rectangle((a[0], a[1], b[0], b[1]), outline=outline, fill=fill, width=width)
    draw.text((80, 20), "R8 临时切分俯视图（按最新区域辨认）", fill="black", font=title)
    for x in range(int(lo_x), int(hi_x)+1):
        draw.line((p(x,lo_z)[0],p(x,lo_z)[1],p(x,hi_z)[0],p(x,hi_z)[1]), fill=(235,235,235), width=1)
    for z in range(int(lo_z), int(hi_z)+1):
        draw.line((p(lo_x,z)[0],p(lo_x,z)[1],p(hi_x,z)[0],p(hi_x,z)[1]), fill=(235,235,235), width=1)
    rect(original, (45,45,45), None, 6)
    rect(sofa, (40,150,70), (220,245,225), 6)
    rect(dining, (40,100,210), (220,232,250), 6)
    def dimension(box, colour, label, horizontal=True):
        if horizontal:
            xa, xb = p(box[0], box[1])[0], p(box[2], box[1])[0]
            y = p(box[1], box[1])[1] + 14
            draw.line((xa, y, xb, y), fill=colour, width=2)
            draw.line((xa, y-7, xa, y+7), fill=colour, width=2)
            draw.line((xb, y-7, xb, y+7), fill=colour, width=2)
            draw.text(((xa+xb)//2-45, y+8), label, fill=colour, font=small)
        else:
            ya, yb = p(box[0], box[1])[1], p(box[0], box[3])[1]
            x = p(box[0], box[0])[0] - 14
            draw.line((x, yb, x, ya), fill=colour, width=2)
            draw.line((x-7, ya, x+7, ya), fill=colour, width=2)
            draw.line((x-7, yb, x+7, yb), fill=colour, width=2)
            draw.text((x-95, (ya+yb)//2-10), label, fill=colour, font=small)
    dimension(original, (45,45,45), "5.67 m", True)
    dimension(original, (45,45,45), "9.94 m", False)
    dimension(sofa, (20,110,45), "3.50 m", True)
    dimension(sofa, (20,110,45), "3.00 m", False)
    dimension(dining, (20,65,170), "3.50 m", True)
    dimension(dining, (20,65,170), "3.70 m", False)
    for point, colour, label in ((camera_sofa,(20,110,45),"相机 S"),(camera_dining,(20,65,170),"相机 D")):
        x,y=p(*point); draw.ellipse((x-10,y-10,x+10,y+10), fill=colour, outline="white", width=2); draw.text((x+14,y-18),label,fill=colour,font=small)
    draw.multiline_text(p(-4.36,1.55), "R8_01\n沙发区", fill=(20,100,45), font=body, spacing=3)
    draw.multiline_text(p(-4.35,-3.7), "R8_02\n餐桌区", fill=(20,65,170), font=body, spacing=3)
    draw.text((80,H-75), "黑框：原始 R8 包围盒（不是沙发区）    绿色：沙发可放置区    蓝色：餐桌可放置区", fill=(30,30,30), font=small)
    draw.text((80,H-45), "临时实验结果；未修改原始 rooms.json、导航网格或审核记录。", fill=(90,90,90), font=small)
    OUT.parent.mkdir(parents=True, exist_ok=True); image.save(OUT); print(OUT)

if __name__ == "__main__":
    main()
