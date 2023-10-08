import uvicorn
from utils.logger import setup_logging


if __name__ == "__main__":
    setup_logging()
    uvicorn.run("apis.api:app", host="0.0.0.0", port=8000, reload=False, log_level="debug", debug=False,
                workers=1, factory=False, loop="asyncio", timeout_keep_alive=120)


