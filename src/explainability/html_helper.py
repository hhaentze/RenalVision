def wrap_svg(svg_data):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Decision Tree</h1>
        <div class="info">
            <p><strong>Instructions:</strong> The decision tree shows the classification logic.</p>
            <ul>
                <li>Each box represents a decision node or prediction</li>
                <li><strong>Orange boxes</strong> = Tumor predictions</li>
                <li><strong>Blue boxes</strong> = Cyst predictions</li>
                <li><strong>Gray boxes</strong> = Uncertain predictions (low confidence)</li>
                <li><strong>Yellow boxes</strong> = Internal decision nodes</li>
                <li>Color intensity indicates purity (darker = more pure)</li>
                <li>Top line shows the decision rule</li>
                <li>gini = measure of impurity (0 = pure)</li>
                <li>value = [tumor_proportion, cyst_proportion]</li>
            </ul>
        </div>
        {svg_data}
    </div>
</body>
</html>
"""
