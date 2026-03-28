import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    deps = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "PIL": "Pillow",
        "flask": "flask",
    }

    optional_deps = {
        "face_recognition": "face_recognition (requires dlib + cmake)",
        "tensorflow": "tensorflow",
    }

    missing = []
    for module, package in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n[ERROR] Missing required dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False

    print("\n[OK] Required dependencies:")
    for module, package in deps.items():
        print(f"  \u2713 {package}")

    print("\n[INFO] Optional dependencies:")
    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"  \u2713 {package}")
        except ImportError:
            print(f"  \u2717 {package} (not installed — some features disabled)")

    return True


def check_models():
    from utils.model_downloader import check_models as _check, models_available

    if models_available():
        print("\n[OK] All models are available")
        return True

    missing = _check()
    print(f"\n[WARNING] {len(missing)} model(s) missing:")
    for key in missing:
        print(f"  - {key}")
    print("\nRun: python main.py --download")
    return False


def download_models():
    from utils.model_downloader import download_all_models

    print("\nDownloading pre-trained models...")
    print("This may take a few minutes depending on your connection.\n")

    def progress_callback(status, progress):
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        print(f"\r  [{bar}] {progress:.0%} - {status}", end="", flush=True)

    success = download_all_models(progress_callback)
    print()

    if success:
        print("\n[OK] All models downloaded successfully!")
    else:
        print("\n[WARNING] Some models failed to download.")
        print("The application will still work with available models.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="FaceVision Pro — Real-Time Facial Recognition System"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download required pre-trained models"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check dependencies and model status"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run the web server on (default: 5000)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU-only mode"
    )

    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("=" * 55)
    print("  FaceVision Pro - Real-Time Facial Recognition System")
    print("=" * 55)

    if args.check:
        check_dependencies()
        check_models()
        return

    if args.download:
        if not check_dependencies():
            sys.exit(1)
        download_models()
        return

    if not check_dependencies():
        sys.exit(1)

    from utils.model_downloader import models_available
    if not models_available():
        print("\n[INFO] Some models are missing. Downloading now...")
        download_models()

    print(f"\n  Starting web server...")
    print(f"  Open your browser at: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    from server import app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
