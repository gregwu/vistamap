"""
FFT cross-axis filter to remove periodic grid artifacts from microscopy images.

Grid-line artifacts (tile seams) produce energy along the entire horizontal and
vertical axes of the FFT. Suppressing those axes — while preserving the DC term —
cleanly removes both horizontal and vertical grid lines.
"""
import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def remove_grid_fft(input_path, output_path, axis_halfwidth=2, dc_keep=8, blur_sigma=1.5):
    """
    Parameters
    ----------
    axis_halfwidth : int
        Half-width (pixels) of the FFT axis lines to suppress.
    dc_keep : int
        Radius around DC to preserve (prevents mean-intensity loss).
    blur_sigma : float
        Gaussian blur on the suppression mask edges to reduce ringing.
    """
    img_pil = Image.open(input_path)
    is_color = img_pil.mode in ("RGB", "RGBA")

    if is_color:
        channels = [np.array(img_pil)[:, :, i].astype(np.float32) for i in range(3)]
    else:
        channels = [np.array(img_pil.convert("L")).astype(np.float32)]

    results = []
    for ch_idx, img in enumerate(channels):
        H, W = img.shape
        cy, cx = H // 2, W // 2

        fshift = np.fft.fftshift(np.fft.fft2(img))

        # Build cross-axis suppression mask
        mask = np.ones((H, W), dtype=np.float32)
        aw = axis_halfwidth
        mask[cy - aw : cy + aw + 1, :] = 0   # horizontal axis -> removes vertical stripes
        mask[:, cx - aw : cx + aw + 1] = 0   # vertical axis   -> removes horizontal stripes
        mask[cy - dc_keep : cy + dc_keep, cx - dc_keep : cx + dc_keep] = 1  # restore DC

        mask = gaussian_filter(mask, sigma=blur_sigma)
        mask[cy - dc_keep : cy + dc_keep, cx - dc_keep : cx + dc_keep] = 1  # re-enforce DC

        img_back = np.fft.ifft2(np.fft.ifftshift(fshift * mask)).real
        results.append(np.clip(img_back, 0, 255).astype(np.uint8))

    if is_color:
        Image.fromarray(np.stack(results, axis=2)).save(output_path)
    else:
        Image.fromarray(results[0]).save(output_path)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove periodic grid artifacts via FFT cross-axis filtering.")
    parser.add_argument("input", help="Input image path (PNG, TIFF, etc.)")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--axis_halfwidth", type=int, default=2,
                        help="Half-width of FFT axis suppression in pixels (default: 2)")
    parser.add_argument("--dc_keep", type=int, default=8,
                        help="DC preservation radius in pixels (default: 8)")
    parser.add_argument("--blur_sigma", type=float, default=1.5,
                        help="Gaussian blur sigma on mask edges to reduce ringing (default: 1.5)")
    args = parser.parse_args()

    print(f"Processing: {args.input}")
    remove_grid_fft(args.input, args.output,
                    axis_halfwidth=args.axis_halfwidth,
                    dc_keep=args.dc_keep,
                    blur_sigma=args.blur_sigma)


if __name__ == "__main__":
    main()
