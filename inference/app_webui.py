from flask import Flask, request, render_template_string, jsonify
from translator import Translator

app = Flask(__name__)
translator = Translator()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>翻译小工具</title>
</head>
<body style="font-family: Arial, sans-serif; padding: 2em;">
    <h2>🌍 英文 ➜ 中文 翻译</h2>
    <form method="post">
        <textarea name="text" rows="5" cols="60" placeholder="请输入英文句子">{{ request.form.text or '' }}</textarea><br><br>
        <input type="submit" value="翻译">
    </form>
    {% if translation %}
        <h3>翻译结果：</h3>
        <div style="border: 1px solid #ccc; padding: 1em; background-color: #f9f9f9;">{{ translation }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST" and "text" in request.form:
        en_text = request.form["text"]
        translation = translator.translate(en_text)
    return render_template_string(HTML_TEMPLATE, translation=translation)

@app.route("/translate", methods=["POST"])
def api_translate():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        result = translator.translate(data["text"])
        return jsonify({"translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800)
