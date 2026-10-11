def write_bild_spheres(path, points, radius, rgb=(0.1, 0.4, 0.9)):
    """
    path: output .bild path
    points: (M,3) array of XYZ in **the same units** as your PDB
    radius: sphere radius (same units)
    rgb: tuple in [0,1]
    """
    r, g, b = rgb
    with open(path, "w") as f:
        f.write(f".color {r:.4f} {g:.4f} {b:.4f}\n")
        for x, y, z in points:
            f.write(f".sphere {x:.4f} {y:.4f} {z:.4f} {radius:.4f}\n")
