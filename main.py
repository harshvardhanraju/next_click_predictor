#!/usr/bin/env python3

import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "Simple Python server",
            "path": self.path,
            "port": os.environ.get("PORT", "not_set"),
            "method": "GET"
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "POST received",
            "path": self.path,
            "port": os.environ.get("PORT", "not_set"),
            "method": "POST"
        }
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"Starting simple server on port {port}")
    server.serve_forever()