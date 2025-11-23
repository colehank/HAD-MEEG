from flask import Flask, render_template, jsonify, request, send_file
from .label_container import ICAData
from pathlib import Path
from dataclasses import asdict
import json

# Get the directory where this file is located
CURRENT_DIR = Path(__file__).parent

# Initialize Flask app with correct template and static folders
app = Flask(
    __name__,
    template_folder=str(CURRENT_DIR / "templates"),
    static_folder=str(CURRENT_DIR / "static"),
)

# Global variables to be set by run_app
all_bids_base = {}
process_file_path = None


def load_processed(process_file: Path):
    """Load processed data from JSON file"""
    if process_file.exists():
        with open(process_file, "r") as f:
            return json.load(f)
    return {}


def save_processed(processed: dict, process_file: Path):
    """Save processed data to JSON file"""
    with open(process_file, "w") as f:
        json.dump(processed, f, indent=2)


def get_remaining_bids(process_file: Path, all_bids_base: dict):
    """Get list of remaining BIDS to process"""
    processed = load_processed(process_file)
    remaining = {
        base: bids for base, bids in all_bids_base.items() if base not in processed
    }
    return remaining


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Get current processing status"""
    processed = load_processed(process_file_path)
    remaining = get_remaining_bids(process_file_path, all_bids_base)

    # Create ordered list of all BIDS basenames
    all_bases = list(all_bids_base.keys())

    total = len(all_bids_base)
    n_processed = len(processed)
    n_remaining = len(remaining)

    # Get current BIDS if available (first unprocessed, or first if all processed)
    current_base = None
    current_index = 0
    if remaining:
        current_base = list(remaining.keys())[0]
        current_index = all_bases.index(current_base)
    elif all_bases:
        current_base = all_bases[0]
        current_index = 0

    return jsonify(
        {
            "total": total,
            "processed": n_processed,
            "remaining": n_remaining,
            "current_base": current_base,
            "current_index": current_index,
            "all_bases": all_bases,
            "progress": f"{n_processed}/{total}",
        }
    )


@app.route("/api/load/<base>")
def load_bids(base):
    """Load ICA data for a specific BIDS"""
    if base not in all_bids_base:
        return jsonify({"error": "BIDS not found"}), 404

    bids = all_bids_base[base]
    data = ICAData(bids)

    # Get component info
    comp_ids = sorted(data.ica_path.comp_pngs.keys(), key=lambda x: int(x))

    # Check if this BIDS was already processed
    processed = load_processed(process_file_path)
    existing_labels = None
    is_processed = False

    if base in processed:
        is_processed = True
        # Load existing manual labels if available
        if processed[base].get("manualed") and processed[base].get("_manual_labels"):
            existing_labels = processed[base]["_manual_labels"]

    return jsonify(
        {
            "base": base,
            "dtype": data.ica_path.dtypr,
            "auto_labels": data.ica_labels.auto_labels,
            "existing_labels": existing_labels,
            "is_processed": is_processed,
            "candidate_labels": data.candidate_labels,
            "comp_ids": comp_ids,
            "n_components": len(comp_ids),
        }
    )


@app.route("/api/image/main/<base>")
def get_main_image(base):
    """Serve main ICA image"""
    if base not in all_bids_base:
        return jsonify({"error": "BIDS not found"}), 404

    bids = all_bids_base[base]
    data = ICAData(bids)

    return send_file(str(data.ica_path.all_png), mimetype="image/png")


@app.route("/api/image/comp/<base>/<comp_id>")
def get_comp_image(base, comp_id):
    """Serve component detail image"""
    if base not in all_bids_base:
        return jsonify({"error": "BIDS not found"}), 404

    bids = all_bids_base[base]
    data = ICAData(bids)

    if comp_id not in data.ica_path.comp_pngs:
        return jsonify({"error": "Component not found"}), 404

    return send_file(str(data.ica_path.comp_pngs[comp_id]), mimetype="image/png")


@app.route("/api/save", methods=["POST"])
def save_labels():
    """Save manual labels for current BIDS"""
    request_data = request.json
    base = request_data.get("base")
    manual_labels = request_data.get("manual_labels")

    if base not in all_bids_base:
        return jsonify({"error": "BIDS not found"}), 404

    bids = all_bids_base[base]
    data = ICAData(bids)

    try:
        # Update manual labels
        data.put_manual_labels(manual_labels)

        # Save to processed file
        processed = load_processed(process_file_path)
        ica_labels = asdict(data.ica_labels)
        processed[base] = ica_labels
        save_processed(processed, process_file_path)

        return jsonify({"success": True, "message": "Labels saved successfully"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/skip", methods=["POST"])
def skip_bids():
    """Skip current BIDS (use auto labels)"""
    request_data = request.json
    base = request_data.get("base")

    if base not in all_bids_base:
        return jsonify({"error": "BIDS not found"}), 404

    bids = all_bids_base[base]
    data = ICAData(bids)

    # Use auto labels as manual labels
    auto_labels = data.ica_labels.auto_labels
    data.put_manual_labels(auto_labels)

    # Save to processed file
    processed = load_processed(process_file_path)
    ica_labels = asdict(data.ica_labels)
    processed[base] = ica_labels
    save_processed(processed, process_file_path)

    return jsonify({"success": True, "message": "Skipped with auto labels"})


def run_app(
    process_file: Path,
    bids_list: list,
    host: str = "0.0.0.0",
    port: int = 5000,
):
    global all_bids_base, process_file_path

    # Convert process_file to Path object if it's a string
    process_file_path = Path(process_file)
    all_bids_base = {bids.basename: bids for bids in bids_list}

    app.run(debug=True, host=host, port=port)
