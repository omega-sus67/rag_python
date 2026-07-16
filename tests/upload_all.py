import asyncio
import httpx
import os

# The PDFs in your trialData directory
PDFS = [
    "trialData/harrypotter.pdf"
]

async def upload_file(client, file_path):
    abs_path = os.path.abspath(file_path)
    print(f"Sending request for: {file_path}")
    
    # POST request to FastAPI
    response = await client.post(
        "http://localhost:8000/upload",
        json={"path": abs_path}
    )
    
    data = response.json()
    print(f"Response for {file_path}: {data.get('message')} | Task ID: {data.get('task_id')}")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Trigger all uploads simultaneously
        await asyncio.gather(*(upload_file(client, pdf) for pdf in PDFS))

if __name__ == "__main__":
    asyncio.run(main())
