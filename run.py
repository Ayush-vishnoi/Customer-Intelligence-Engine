"""
run.py — Launch the Flask app locally.
Usage:
    python run.py              # port 5050
    python run.py --port 8080
    python run.py --prod       # disable debug mode
"""

import sys, os, argparse, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def ensure_processed():
    required = [
        os.path.join(ROOT, "data", "processed", "customer_segments.csv"),
        os.path.join(ROOT, "data", "processed", "monthly_revenue.csv"),
        os.path.join(ROOT, "data", "processed", "top_products.csv"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("[run] Missing processed data. Run 'python main.py' first.")
        for f in missing:
            print(f"  missing: {f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    ensure_processed()

    spec = importlib.util.spec_from_file_location("app.app", os.path.join(ROOT, "app", "app.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print(f"\n  CustomerIQ — http://localhost:{args.port}\n")
    mod.app.run(debug=not args.prod, port=args.port, host="0.0.0.0", use_reloader=False)
