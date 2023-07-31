from GraphTranslation.services.base_service import BaseServiceSingleton


class TranslationService(BaseServiceSingleton):
    def __init__(self):
        super(TranslationService, self).__init__()
