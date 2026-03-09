"""Quick benchmark: switch to webcam 4 turbo and measure FPS via HTTP."""
import requests, json, time, asyncio
try:
    import websockets
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


async def main():
    uri = "ws://localhost:8000/ws"
    print("Connecting...")
    ws = await websockets.connect(uri, ping_interval=None, ping_timeout=None,
                                   open_timeout=30, close_timeout=2)
    print("Connected. Sending set_source...")
    await ws.send(json.dumps({"action": "set_source", "source": "4", "label": "Webcam 4"}))

    # Drain messages until we get ack or timeout
    t0 = time.time()
    got_ack = False
    while time.time() - t0 < 25:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            msg = json.loads(raw)
            if msg.get("type") == "ack" and msg.get("action") == "set_source":
                print("set_source:", msg)
                got_ack = True
                break
        except asyncio.TimeoutError:
            print("  waiting for ack...")
            continue
    
    if not got_ack:
        print("WARN: No ack received, continuing anyway")

    # Set turbo
    print("Setting turbo profile...")
    await ws.send(json.dumps({"action": "set_perf_profile", "profile": "turbo"}))
    time.sleep(1)  # let it settle

    # Close WS and just poll stats via HTTP
    await ws.close()
    print("Monitoring FPS via HTTP...")
    for i in range(10):
        time.sleep(2)
        try:
            r = requests.get("http://localhost:8000/stats", timeout=3)
            d = r.json()
            print(f"  [{i}] FPS={d['fps']}  source_fps={d['source_fps']}  "
                  f"imgsz={d['imgsz']}  personas={d['personas']}  "
                  f"device={d['device']}  mode={d['mode']}")
        except Exception as e:
            print(f"  [{i}] Error: {e}")


asyncio.run(main())
