import os
import sys

# Wrapper script to run distributed training from the root
if __name__ == "__main__":
    script_path = os.path.join("pipelines", "train", "distributed.py")
    if not os.path.exists(script_path):
        print(f"Error: Internal training script not found at {script_path}")
        sys.exit(1)
        
    # Pass all arguments to the main script
    import subprocess
    cmd = [sys.executable, script_path] + sys.argv[1:]
    subprocess.run(cmd)
