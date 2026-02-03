from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import os  # noqa: E402
import cv2  # noqa: E402

SRC = r"screen.png"
OUTDIR = r"templates"

os.makedirs(OUTDIR, exist_ok=True)

img = cv2.imread(SRC)
if img is None:
    raise SystemExit("Не нашёл screen.png рядом со скриптом")

r = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
x, y, w, h = map(int, r)
crop = img[y : y + h, x : x + w]

name = input("Имя шаблона (например tpl_play): ").strip()
path = os.path.join(OUTDIR, f"{name}.png")
cv2.imwrite(path, crop)
print("Saved:", path)
cv2.destroyAllWindows()
