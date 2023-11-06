from fastapi import FastAPI
#from fastapi.staticfiles import StaticFiles
import os

# from apis.routes.graph_translate import GraphTranslateRoute
from apis.routes.translation import TranslateRoute
#from apis.routes.texttospeech import SpeakRoute
from apis.routes.addword import addWord
from apis.routes.update import updateWord
from apis.routes.changeCorpus import changeCorpus
from apis.routes.deleteword import deleteWord
# from fastapi.middleware.cors import CORSMiddleware
from apis.routes.reverseTranslation import reverseTranslate
from starlette.middleware.cors import CORSMiddleware
from GraphTranslation.common.languages import Languages
from GraphTranslation.config.config import Config
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
app = FastAPI()

# app.add_middleware(HTTPSRedirectMiddleware)
import yaml

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
# read area.yaml file to check for current area. If GiaLai then keep
# else delete current area.yaml file
delete = False
area = 'GiaLai'
if os.path.exists("data/cache/info.yaml"):
    with open('data/cache/info.yaml', 'r+') as f:
        # if the "area" field is not GiaLai then delete
        data = yaml.safe_load(f)
        area = data.get('area', None)
        if area != 'GiaLai':
            delete = True
        
if delete:
    if os.path.exists("data/cache/info.yaml"):
        os.remove("data/cache/info.yaml")

    if os.path.exists("data/cache/graph.json"):
        os.remove("data/cache/graph.json")
              
app.include_router(TranslateRoute("GiaLai").router)
#app.include_router(SpeakRoute().router)
app.include_router(addWord("GiaLai").router)
app.include_router(updateWord("GiaLai").router)
app.include_router(changeCorpus("GiaLai").router)
app.include_router(deleteWord("GiaLai").router)
app.include_router(reverseTranslate("GiaLai").router)

yaml.dump({"area": "GiaLai"}, open("data/cache/info.yaml", "w"))
# append SRC into info.yaml
yaml.dump({"SRC": Languages.SRC}, open("data/cache/info.yaml", "a"))
# append DST into info.yaml
yaml.dump({"DST": Languages.DST}, open("data/cache/info.yaml", "a"))


datapath = "data/" + area + '/'
# count number of sentences in train, valid, test of the area
with open(datapath + Config.src_monolingual_paths[0], "r", encoding='utf-8') as f1:
    src_train_count = len(f1.readlines())
with open(datapath + Config.src_monolingual_paths[1], "r", encoding='utf-8') as f2:
    src_valid_count = len(f2.readlines())
with open(datapath + Config.src_mono_test_paths[0], "r", encoding='utf-8') as f3:
    src_test_count = len(f3.readlines())
with open(datapath + Config.dst_monolingual_paths[0], "r", encoding='utf-8') as f4:
    dst_train_count = len(f4.readlines())
with open(datapath + Config.dst_monolingual_paths[1], "r", encoding='utf-8') as f5:
    dst_valid_count = len(f5.readlines())
with open(datapath + Config.dst_mono_test_paths[0], "r", encoding='utf-8') as f6:
    dst_test_count = len(f6.readlines())

with open("data/cache/info.yaml", "a") as f:
    yaml.dump({
        "src_train": src_train_count,
        "src_valid": src_valid_count,
        "src_test": src_test_count,
        "dst_train": dst_train_count,
        "dst_valid": dst_valid_count,
        "dst_test": dst_test_count
    }, f)

with open("data/cache/info.yaml", "r") as f:
    print(f.read())

