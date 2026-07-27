import sys
import subprocess

def main():
    print("Redirecting to modernized baseline training pipeline (src/train_baseline.py)...")
    result = subprocess.run([sys.executable, "src/train_baseline.py"])
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()