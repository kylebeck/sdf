from sdf import *
import time

def main():
    print("Building procedural model (sphere & box)...")
    f = sphere(1) | box(1).translate((0.5, 0, 0))
    
    out_path = 'neural_out.obj'
    print(f"Baking and exporting neural quad skeleton to {out_path}...")
    
    t0 = time.time()
    # Use the new neural_quad method
    f.save(out_path, method='neural_quad', bounds=((-2, -2, -2), (2, 2, 2)), verbose=True)
    t1 = time.time()
    
    print(f"Total time elapsed: {t1-t0:.2f}s")
    print("Done! Check neural_out.obj")

if __name__ == '__main__':
    main()
