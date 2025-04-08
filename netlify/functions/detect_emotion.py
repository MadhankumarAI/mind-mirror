import base64
from io import BytesIO
from PIL import Image
from fer import FER
import json

def handler(event, context):
    try:
        body = json.loads(event['body'])
        image_data = body['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        detector = FER()
        result = detector.detect_emotions(image)
        emotion = None
        if result and 'emotions' in result[0]:
            emotions = result[0]['emotions']
            emotion = max(emotions, key=emotions.get)

        return {
            "statusCode": 200,
            "headers": { "Access-Control-Allow-Origin": "*" },
            "body": json.dumps({ "emotion": emotion })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": { "Access-Control-Allow-Origin": "*" },
            "body": json.dumps({ "error": str(e) })
        }
