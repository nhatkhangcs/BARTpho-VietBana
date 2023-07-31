import setproctitle
setproctitle.setproctitle('bahnar_tts_nmt')
import uvicorn
from utils.logger import setup_logging

def main():
    setup_logging()
    uvicorn.run("apis.api:app", host="192.168.1.5", port=8000, reload=False, log_level="debug",
                workers=1, factory=False, loop="asyncio", timeout_keep_alive=0,
                )
    

if __name__ == "__main__":
    main()
