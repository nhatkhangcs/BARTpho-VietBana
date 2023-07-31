from fastapi import FastAPI

from GraphTranslation.apis.routes.graph_translate import GraphTranslateRoute


app = FastAPI()
app.include_router(GraphTranslateRoute().router)
