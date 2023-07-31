from tqdm import tqdm

from GraphTranslation.pipeline.translation import TranslationPipeline
from custom_dataset.vi_ba_aligned_dataset import ViBaDataset


phrase_chunking = TranslationPipeline()
phrase_chunking.add_check_valid_anchor_func(ViBaDataset.check_valid_anchor)

dataset = ViBaDataset.load_data(data_folder="data/all", mode="train", return_index=True)
print(len(dataset))
chunk_data_vi = []
chunk_data_ba = []
for i, (data_source, vi, ba) in enumerate(tqdm(dataset)):
    if vi.count(" ") < 5:
        continue
    mapped_chunks, not_mapped_chunks = phrase_chunking.extract_chunks(vi, ba)
    o_vi = []
    o_ba = []
    o_source = []
    for vi_chunk, ba_chunk in mapped_chunks:
        o_vi.append(vi_chunk.text.strip())
        o_ba.append(ba_chunk.text.strip())
        o_source.append(f"{vi}\t=======\t{ba}")

    open("scripts/chunk_data.vi", "a+", encoding="utf8").write("\n".join(o_vi))
    open("scripts/chunk_data.ba", "a+", encoding="utf8").write("\n".join(o_ba))
    open("scripts/source.txt", "a+", encoding="utf8").write("\n".join(o_source))
    # print(vi, ba, mapped_chunks)
    # c += 1
    # if c == 10:
    #     break

