import requests
import unittest

class Test(unittest.TestCase):
    def addwordtest(self):
        print("Add word API...")
        res = requests.post("http://localhost:6379/addword/vi_ba", json={"word": ["tôi"], "translation": ["I"]})
        self.assertEqual(res.status_code, 200)

    def deletewordtest(self):
        print("Delete word API...")
        res = requests.post("http://localhost:6379/deleteword/vi_ba", json={"word": "tôi"})
        self.assertEqual(res.status_code, 200)

    def updatewordtest(self):
        print("Update word API...")
        res = requests.post("http://localhost:6379/updateword/vi_ba", json={"word": "holy", "translation": "smoke"})
        self.assertEqual(res.status_code, 200)

    def changecorpus(self):
        print("Change corpus API...")
        res = requests.post("http://localhost:6379/changeCorpus/vi_ba", json={"area": "GiaLai"})
        self.assertEqual(res.status_code, 200)

    def translate(self):
        print("Translate API...")
        res = requests.post("http://localhost:6379/translate/vi_ba", json={"text": "holy"})
        self.assertEqual(res.status_code, 200)

        # check if the translation is correct
        self.assertEqual(res.json(), {"text": "holy", "translation": "smoke"})

if __name__ == '__main__':
    unittest.main()