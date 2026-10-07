"""Single source of truth for the calibration target, and its printable PDF.

Every other tool imports BOARD from here, so the geometry that gets detected can
never disagree with the geometry that got printed. The generated page carries
its own parameters and a 100 mm scale bar, so a sheet found on a desk months
from now still identifies itself and can be checked for print scaling.

Layout is 7x10 squares at 25 mm = 175 x 250 mm on A4 portrait. That leaves
17.5 mm side margins and 36 mm at the foot for the reference block. The wider
8x11 board would fill the page to within 5 mm of the edge, which most printers
cannot reach and which leaves no white quiet zone for the edge ArUco markers.

Run with:  uv run board.py            -> charuco_A4.pdf
"""

import argparse

import cv2
import numpy as np
from PIL import Image

# 20 px/mm = 508 dpi keeps every millimetre an exact integer number of pixels,
# so the printed geometry matches the nominal geometry exactly.
PX_PER_MM = 20
DPI = PX_PER_MM * 25.4

BOARD = {
    "squares_x": 7,
    "squares_y": 10,
    "square_mm": 25.0,
    "marker_mm": 18.75,     # 0.75 x square, the usual ratio
    "dictionary": "DICT_4X4_50",   # 35 markers needed, 50 available
}

A4_W_MM, A4_H_MM = 210, 297

# --- far-range target ------------------------------------------------------
# The calibration board's 18.75 mm markers fall under 3 px per module past
# about 1.3 m, so depth cannot be validated beyond that. Bigger markers reach
# much further: a 2x2 grid of 85 mm markers fits A4 and stays decodable to
# roughly 5 m, while 16 corners still give a solid pose.
#
# IDs start at 40 so they can never be confused with the calibration board,
# which uses 0-34 of the same dictionary.
FAR_BOARD = {
    "markers_x": 2,
    "markers_y": 2,
    "marker_mm": 85.0,
    "separation_mm": 20.0,
    "dictionary": "DICT_4X4_50",
    "first_id": 40,
}


def make_board(square_mm=None):
    """The cv2 board object. square_mm overrides the nominal size with the
    value you measured on the actual print, which is what calibration must use.
    """
    s = (square_mm or BOARD["square_mm"]) / 1000.0
    # Keep the marker/square ratio when the square size is corrected.
    m = s * (BOARD["marker_mm"] / BOARD["square_mm"])
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, BOARD["dictionary"]))
    return cv2.aruco.CharucoBoard(
        (BOARD["squares_x"], BOARD["squares_y"]), s, m, adict)


def n_corners():
    return (BOARD["squares_x"] - 1) * (BOARD["squares_y"] - 1)


def render_page():
    """Compose the A4 page: board, reference block, scale bar. Returns uint8."""
    page_w = A4_W_MM * PX_PER_MM
    page_h = A4_H_MM * PX_PER_MM
    page = np.full((page_h, page_w), 255, np.uint8)

    sq_px = int(BOARD["square_mm"] * PX_PER_MM)
    bw = BOARD["squares_x"] * sq_px
    bh = BOARD["squares_y"] * sq_px
    board_img = make_board().generateImage((bw, bh), marginSize=0)

    x0 = (page_w - bw) // 2
    y0 = 11 * PX_PER_MM
    page[y0:y0 + bh, x0:x0 + bw] = board_img

    # --- reference block -----------------------------------------------------
    ty = y0 + bh + 9 * PX_PER_MM
    line1 = (f"ChArUco {BOARD['squares_x']}x{BOARD['squares_y']}  |  "
             f"square {BOARD['square_mm']:.2f} mm  |  "
             f"marker {BOARD['marker_mm']:.2f} mm")
    line2 = (f"{BOARD['dictionary']}  |  {n_corners()} corners  |  "
             f"A4 portrait, print at 100% (no 'fit to page')")
    cv2.putText(page, line1, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, 0, 4, cv2.LINE_AA)
    cv2.putText(page, line2, (x0, ty + 4 * PX_PER_MM), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, 0, 3, cv2.LINE_AA)

    # --- 100 mm scale bar ----------------------------------------------------
    # The whole point: printers silently rescale, and a wrong square size
    # becomes a wrong baseline and a wrong depth scale that nothing downstream
    # can detect. Measuring this bar catches it in ten seconds.
    by = ty + 12 * PX_PER_MM
    bar_len = 100 * PX_PER_MM
    cv2.line(page, (x0, by), (x0 + bar_len, by), 0, 4)
    for mm in range(0, 101, 10):
        x = x0 + mm * PX_PER_MM
        tick = 3 * PX_PER_MM if mm % 50 == 0 else 2 * PX_PER_MM
        cv2.line(page, (x, by), (x, by - tick), 0, 4)
    cv2.putText(page, "measure me: must be exactly 100.0 mm",
                (x0 + bar_len + 3 * PX_PER_MM, by),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3, cv2.LINE_AA)
    cv2.putText(page,
                "If it is not, measure one square with calipers and pass that "
                "value to calibration.",
                (x0, by + 5 * PX_PER_MM),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    return page


def save_pdf(path):
    """Render the page and write it out as a PDF at the given path."""
    page = render_page()
    Image.fromarray(page).save(path, "PDF", resolution=DPI)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="charuco_A4.pdf")
    args = p.parse_args()

    save_pdf(args.out)

    page = render_page()
    h, w = page.shape
    print(f"Wrote {args.out}")
    print(f"  page   {w}x{h} px at {DPI:.0f} dpi = "
          f"{w / PX_PER_MM:.1f} x {h / PX_PER_MM:.1f} mm")
    print(f"  board  {BOARD['squares_x']}x{BOARD['squares_y']} @ "
          f"{BOARD['square_mm']} mm = "
          f"{BOARD['squares_x'] * BOARD['square_mm']:.0f} x "
          f"{BOARD['squares_y'] * BOARD['square_mm']:.0f} mm, "
          f"{n_corners()} corners")
    print("\nPrint at 100% scale on matte paper, then mount it flat on foam "
          "board or glass.")


if __name__ == "__main__":
    main()


def make_far_board(marker_mm=None, separation_mm=None):
    """ArUco GridBoard for validating depth beyond the calibration board's
    ~1.3 m detection limit."""
    m = (marker_mm or FAR_BOARD["marker_mm"]) / 1000.0
    sep = (separation_mm or FAR_BOARD["separation_mm"]) / 1000.0
    adict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, FAR_BOARD["dictionary"]))
    ids = np.arange(FAR_BOARD["first_id"],
                    FAR_BOARD["first_id"]
                    + FAR_BOARD["markers_x"] * FAR_BOARD["markers_y"])
    return cv2.aruco.GridBoard(
        (FAR_BOARD["markers_x"], FAR_BOARD["markers_y"]), m, sep, adict, ids)


def far_extent_mm():
    b = FAR_BOARD
    return (b["markers_x"] * b["marker_mm"] + (b["markers_x"] - 1) * b["separation_mm"],
            b["markers_y"] * b["marker_mm"] + (b["markers_y"] - 1) * b["separation_mm"])


def render_far_page():
    """A4 page with the far-range grid, its parameters and a 100 mm scale bar."""
    page_w, page_h = A4_W_MM * PX_PER_MM, A4_H_MM * PX_PER_MM
    page = np.full((page_h, page_w), 255, np.uint8)

    bw_mm, bh_mm = far_extent_mm()
    bw, bh = int(bw_mm * PX_PER_MM), int(bh_mm * PX_PER_MM)
    # A quiet zone is REQUIRED: markers touching the image edge are not
    # detected at all, which is how the first attempt found 0 of 4.
    margin = int(8 * PX_PER_MM)
    img = make_far_board().generateImage((bw + 2 * margin, bh + 2 * margin),
                                         marginSize=margin)
    x0 = (page_w - img.shape[1]) // 2
    y0 = 14 * PX_PER_MM
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    tx = x0 + margin
    ty = y0 + img.shape[0] + 10 * PX_PER_MM
    b = FAR_BOARD
    cv2.putText(page, f"ArUco grid {b['markers_x']}x{b['markers_y']}  |  "
                      f"marker {b['marker_mm']:.1f} mm  |  "
                      f"gap {b['separation_mm']:.1f} mm",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 0, 4, cv2.LINE_AA)
    cv2.putText(page, f"{b['dictionary']} ids "
                      f"{b['first_id']}-{b['first_id'] + 3}  |  "
                      f"FAR-RANGE validation target, good to about 5 m  |  "
                      f"print at 100%",
                (tx, ty + 4 * PX_PER_MM), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, 0, 3, cv2.LINE_AA)

    by = ty + 12 * PX_PER_MM
    bar = 100 * PX_PER_MM
    cv2.line(page, (tx, by), (tx + bar, by), 0, 4)
    for mm in range(0, 101, 10):
        x = tx + mm * PX_PER_MM
        cv2.line(page, (x, by), (x, by - (3 if mm % 50 == 0 else 2) * PX_PER_MM), 0, 4)
    cv2.putText(page, "measure me: must be exactly 100.0 mm",
                (tx + bar + 3 * PX_PER_MM, by),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3, cv2.LINE_AA)
    return page


def save_far_pdf(path):
    from PIL import Image
    Image.fromarray(render_far_page()).save(path, "PDF", resolution=DPI)
