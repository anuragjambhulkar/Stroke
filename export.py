import gradio as gr
from app import build_interface  # or from test import build_interface if you want test.py

def export_app():
    app = build_interface()
    print("🚀 Exporting Gradio app to ./build ...")

    # Use the internal exporter API (equivalent to `gradio export`)
    from gradio.exporter import export_app as internal_export
    internal_export(app, out_dir="build")

    print("✅ Export complete! Files are in ./build")

if __name__ == "__main__":
    export_app()
