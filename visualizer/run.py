#!/usr/bin/env python
"""
Codon Encoder Visualizer Launcher

Usage:
    python run.py                           # Use default model
    python run.py path/to/model.pt          # Use specific model
    python run.py --port 8080               # Custom port
    python run.py model.pt --host 0.0.0.0   # Expose to network
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Codon Encoder Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Default model
  python run.py ../other_model.pt         # Different model
  python run.py --export data.json        # Export visualization data
  python run.py --test                    # Run model tests only
        """
    )

    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to .pt model file (default: codon_encoder.pt)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export visualization data to JSON file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run model loader test without starting server"
    )

    args = parser.parse_args()

    # Resolve model path
    model_path = None
    if args.model_path:
        model_path = Path(args.model_path).resolve()
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
        print(f"Using model: {model_path}")

    # Export mode
    if args.export:
        from config import Config
        from model_loader import ModelLoader

        if model_path:
            Config.set_model_path(model_path)

        loader = ModelLoader().load()
        json_data = loader.to_json()

        with open(args.export, 'w') as f:
            f.write(json_data)
        print(f"Exported to: {args.export}")
        return

    # Test mode
    if args.test:
        from config import Config
        if model_path:
            Config.set_model_path(model_path)

        from model_loader import main as test_main
        test_main()
        return

    # Start server - pass model_path to run()
    from server import run
    run(host=args.host, port=args.port, model_path=model_path)


if __name__ == "__main__":
    main()
