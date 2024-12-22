#!/bin/bash

URL="http://127.0.0.1:8000/predict"
JSON_PAYLOAD='{"text": "I love this movie !"}'


curl -X POST "$URL" \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD"
