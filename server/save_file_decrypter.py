import websockets
import asyncio
import json
import jsondiff

# async def main():
#     await websockets.serve(_on_connect, "localhost", 8080)
#     await asyncio.Future()


# async def _on_connect(websocket, path):
#     print("Connected to client")
#     text = await websocket.recv()
#     file = open("save_file.json", "w")
#     file.write(text)
#     file.close()
#     await asyncio.Future()

# asyncio.run(main())

json1 = json.loads(open("save_file.json", "r").read())
json2 = json.loads(open("Resource\completed_save.json", "r").read())

jsondiff.diff(json1, json2)