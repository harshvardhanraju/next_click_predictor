#!/usr/bin/env python3

import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        
        response = {
            "message": "Simple Python server working!",
            "path": self.path,
            "port": os.environ.get("PORT", "not_set"),
            "method": "GET",
            "status": "healthy"
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        
        response = {
            "message": "POST received successfully",
            "path": self.path,
            "port": os.environ.get("PORT", "not_set"),
            "method": "POST",
            "status": "working"
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"=== Railway Deployment Test ===")
    print(f"PORT from environment: {os.environ.get('PORT', 'NOT_SET')}")
    print(f"Starting server on 0.0.0.0:{port}")
    
    try:
        server = HTTPServer(('0.0.0.0', port), SimpleHandler)
        print(f"✅ Server started successfully on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        raise