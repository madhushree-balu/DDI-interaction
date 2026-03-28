from flask import Flask, render_template, request, jsonify
from main import search_brand_name, analyze_interactions_with_context, get_ingredients_by_brand_name

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '')
    if not q:
        return jsonify([])
    try:
        results = search_brand_name(q, limit=10)
        enhanced_results = [
            {"brand": b, "ingredients": get_ingredients_by_brand_name(b)}
            for b in results
        ]
        return jsonify(enhanced_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid payload'}), 400
        
    brand_names = data.get('brand_names', [])
    patient_data = data.get('patient_data', None)
    
    if not brand_names:
        return jsonify({'error': 'No brand names provided'}), 400
        
    try:
        # Run the full PolyGuard pipeline
        report = analyze_interactions_with_context(
            brand_names=brand_names,
            patient_data=patient_data,
            save_report=None,
            explain=True
        )
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
