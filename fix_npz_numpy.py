"""
Convert a numpy .npz policy file saved with numpy 2.x to be compatible with numpy 1.x.
Usage:
    python convert_npz_compat.py <input.npz> [output.npz]

If no output path is given, saves as <input>_compat.npz in the same directory.
"""

import sys
import os
import numpy as np


def convert(input_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_compat" + ext

    print(f"Loading: {input_path}")
    data = np.load(input_path, allow_pickle=True)

    arrays = {}
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        arrays[key] = arr

    np.savez(output_path, **arrays)
    print(f"Saved:   {output_path}")

    # Verify
    check = np.load(output_path, allow_pickle=True)
    for key in check.files:
        arr = check[key]
        print(f"  [verify] {key}: shape={arr.shape}, dtype={arr.dtype}")
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    convert(input_path, output_path)