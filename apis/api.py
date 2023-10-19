from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles
#import os

# from apis.routes.graph_translate import GraphTranslateRoute
from apis.routes.translation import TranslateRoute
#from apis.routes.texttospeech import SpeakRoute
#from apis.routes.addword import addWord
#from apis.routes.update import updateWord
from apis.routes.changeCorpus import changeCorpus
# from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.cors import CORSMiddleware

# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
app = FastAPI()

# app.add_middleware(HTTPSRedirectMiddleware)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#app.mount("/to-speech", StaticFiles(directory=os.path.abspath("to-speech")), name="to-speech")

# app.include_router(GraphTranslateRoute().router)
app.include_router(TranslateRoute("GiaLai").router)
#app.include_router(SpeakRoute().router)
#app.include_router(addWord("BinhDinh").router)
#app.include_router(updateWord().router)
app.include_router(changeCorpus("GiaLai").router)