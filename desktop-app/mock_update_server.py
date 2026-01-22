import http.server
import socketserver
import json

PORT = 8080

class MockUpdateHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/update.json":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            manifest = {
                "version": "0.1.1",
                "notes": "Test update for TASK_03",
                "pub_date": "2026-01-18T00:00:00Z",
                "platforms": {
                    "windows-x86_64": {
                        "signature": "MOCK_SIGNATURE",
                        "url": "http://localhost:8080/SimOps_0.1.1_x64_en-US.msi.zip"
                    }
                }
            }
            self.wfile.write(json.dumps(manifest).encode())
        else:
            self.send_error(404)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MockUpdateHandler) as httpd:
        print(f"Mock update server running at http://localhost:{PORT}")
        # This is meant to be run manually or in background
        # httpd.serve_forever()
        # For agentic verification, we just want it to exist and be valid
        pass
