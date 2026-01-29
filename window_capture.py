import pygetwindow as gw

def get_window_region(title_contains: str):
    wins = [w for w in gw.getAllWindows() if title_contains.lower() in w.title.lower()]
    if not wins:
        raise RuntimeError(f"Не нашёл окно с '{title_contains}'. Проверь заголовок окна игры.")
    w = wins[0]
    # координаты внешнего окна
    left, top = w.left, w.top
    width, height = w.width, w.height

    # Иногда рамки/заголовок дают сдвиг. Если видишь рамку в захвате — подправь эти значения:
    border_x = 8
    titlebar_y = 32
    return dict(
        left=left + border_x,
        top=top + titlebar_y,
        width=width - 2 * border_x,
        height=height - titlebar_y - border_x,
    )
