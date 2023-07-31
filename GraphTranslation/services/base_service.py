import os
import logging
from tqdm import tqdm
import pickle
from queue import Queue

from objects.singleton import Singleton
from GraphTranslation.config.config import Config
from GraphTranslation.utils.utils import generate_id


class BaseService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()

    def make_request(self, request_func, key=None, **kwargs):
        result = None
        while result is None:
            # try:
            result = request_func(**kwargs)
            # except Exception as e:
            #     logger = logging.getLogger(self.__class__.__name__)
            #     logger.error(f"Cannot make request. Error: {e}. Func: {request_func}. Args: {kwargs}")
        return result


class CacheBaseService(BaseService):
    def __init__(self):
        super(CacheBaseService, self).__init__()
        self.cache = {}
        self.cache_queue = Queue(maxsize=self.config.cache_size)

    def add_item(self, key, item):
        self.cache[key] = item
        self.cache_queue.put(key)
        if self.cache_queue.qsize() == self.config.cache_size:
            remove_key = self.cache_queue.get()
            if remove_key in self.cache:
                del self.cache[remove_key]

    def get_item(self, key):
        if key in self.cache:
            return self.cache[key]

    def make_request(self, request_func, key=None, **kwargs):
        if key is None:
            return super().make_request(request_func, **kwargs)
        else:
            output = self.get_item(key)
            if output is not None:
                return output
            else:
                output = super().make_request(request_func, **kwargs)
                self.add_item(key, output)
                return output


class CacheFileService(CacheBaseService):
    def __init__(self):
        super(CacheFileService, self).__init__()
        self.cache_folder = f".cache/{self.__class__.__name__}"
        os.makedirs(self.cache_folder, exist_ok=True)
        # self.load_cache_file()

    def load_cache_file(self):
        for file_name in tqdm(os.listdir(self.cache_folder), desc=f"LOAD FROM DISK CACHE - {self.cache_folder}"):
            file_path = os.path.join(self.cache_folder, file_name)
            cache_data = self.load_from_disk(file_path=file_path)
            self.cache[cache_data["key"]] = cache_data["data"]
            self.cache_queue.put(cache_data["key"])
            if self.cache_queue.qsize() == self.config.cache_size - 1:
                break

    def get_file_path(self, key):
        data_id = generate_id(key)
        file_path = os.path.join(self.cache_folder, f"{data_id}.dat")
        return file_path

    def save_data(self, key, data):
        json_data = {"key": key, "data": data}
        file_path = self.get_file_path(key)
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as handle:
                pickle.dump(json_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_disk(self, key=None, file_path=None):
        if file_path is None:
            file_path = self.get_file_path(key)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)
            return data

    def get_item(self, key):
        output = super().get_item(key)
        if output is not None:
            self.logger.debug("GET FROM RAM CACHE")
            return output
        output = self.load_from_disk(key)
        if output is not None:
            self.logger.debug("GET FROM DISK CACHE")
            output = output["data"]
            self.add_item(key, output)
            self.logger.debug("STORE FROM DISK CACHE TO RAM CACHE")
            return output

    def add_item(self, key, item):
        self.cache[key] = item
        self.cache_queue.put(key)
        self.save_data(key, item)
        if self.cache_queue.qsize() == self.config.cache_size:
            remove_key = self.cache_queue.get()
            if remove_key in self.cache:
                del self.cache[remove_key]


class BaseServiceSingletonWithCache(CacheFileService, metaclass=Singleton):
    pass


class BaseServiceSingleton(BaseService, metaclass=Singleton):
    pass


class BaseSingleton(metaclass=Singleton):
    pass
