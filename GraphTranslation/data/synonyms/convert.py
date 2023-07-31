import json

data = json.load(open("vi_syn_data.json", "r", encoding="utf8"))

json.dump(data, open("vi_syn_data_1.json", "w", encoding="utf8"), ensure_ascii=False, indent=4)
