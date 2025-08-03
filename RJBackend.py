import sys
import json

def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = {"echo": request.get("message", "")}
            print(json.dumps(response), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()