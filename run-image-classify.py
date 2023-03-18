import requests

with open('Image-Classification/testset/val/img1.jpeg', 'rb') as f:
    res = requests.post('http://127.0.0.1:8000/api/run',
                files={'upload': f},
                )
print(res.content)