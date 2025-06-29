<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result - Hematovision</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
            padding-top: 40px;
            text-align: center;
        }
        .container {
            max-width: 700px;
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin: auto;
        }
        .header {
            background-color: #f44336;
            color: white;
            padding: 15px;
            font-size: 24px;
            font-weight: bold;
            border-radius: 10px 10px 0 0;
        }
        .blood-image {
            max-width: 300px;
            border-radius: 10px;
            margin: 20px auto;
        }
        .btn-custom {
            background-color: #f44336;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            margin-top: 20px;
        }
        .btn-custom:hover {
            background-color: #d93025;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">Prediction Result</div>
        <br>
        <h5><strong>Predicted Class:</strong> {{ class_label }}</h5>
        
        <img src="data:image/png;base64,{{ img_data }}" alt="Blood Cell Image" class="blood-image">

        <br>
        <a href="/" class="btn btn-custom">Upload Another Image</a>
    </div>
</body>
</html>

