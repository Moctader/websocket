import asyncio
import websockets
import json

async def listen():
    uri = "wss://ws.eodhistoricaldata.com/ws/us?api_token=demo"
    try:
        async with websockets.connect(uri) as websocket:
            # Subscribe to MSFT ticker
            subscribe_message = json.dumps({
                "action": "subscribe",
                "symbols": "AAPL"
            })
            await websocket.send(subscribe_message)
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received data: {data}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

asyncio.get_event_loop().run_until_complete(listen())