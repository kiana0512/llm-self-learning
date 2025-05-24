from flask import Flask, request, jsonify
from translator import Translator

app = Flask(__name__)
translator = Translator()

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        result = translator.translate(data["text"])
        return jsonify({"translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "<h2>✅ 模型翻译服务运行中，请 POST 到 /translate</h2>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800)
