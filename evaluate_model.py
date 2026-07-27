import sys
import subprocess

def main():
    print("Redirecting to modernized evaluation pipeline (src/evaluate.py)...")
    result = subprocess.run([sys.executable, "src/evaluate.py"])
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()