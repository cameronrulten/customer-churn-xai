import uvicorn

def main():
    # You can parameterize host/port via env vars later if you like
    uvicorn.run("churn_xai.api.main:app", host="0.0.0.0", port=8000, reload=True)
