import setproctitle
setproctitle.setproctitle('bahnar_tts_nmt')
import uvicorn
from utils.logger import setup_logging

HUY = "192.168.1.153"
HUY_5G = "192.168.31.76"
HCMUT1 = "10.128.147.168"
HCMUT2 = "10.130.193.88"
KHANG_5G = "192.168.1.4"
PHONG_5G = "192.168.1.9"


def main():
    setup_logging()
    uvicorn.run("apis.api:app", host=PHONG_5G, port=6379, reload=False, log_level="debug",
                workers=1, factory=False, loop="asyncio", timeout_keep_alive=0,
                )
    

if __name__ == "__main__":
    main()
