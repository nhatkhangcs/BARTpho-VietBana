import setproctitle
setproctitle.setproctitle('bahnar_tts_nmt')
import uvicorn
from utils.logger import setup_logging

NHATKHANG1 = "192.168.1.3"
NHATKHANG5G = ""
ROOM922 = ""
HCMUT1 = ""
HCMUT2 = ""


def main():
    setup_logging()
    uvicorn.run("apis.api:app", host=NHATKHANG1, port=6379, reload=False, log_level="debug",
                workers=1, factory=False, loop="asyncio", timeout_keep_alive=0,
                )
    

if __name__ == "__main__":
    main()
