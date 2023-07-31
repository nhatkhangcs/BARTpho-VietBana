import collections
import json
from random import shuffle

import datasets


_DESCRIPTION = """\
Preprocessed Dataset Vietnamese Spelling Autocorrection.
"""



class ViBaDatasetConfig(datasets.BuilderConfig):

    def __init__(self, language_pair=(None, None), **kwargs):

        description = ("Translation dataset from %s to %s") % (language_pair[0], language_pair[1])
        super(ViBaDatasetConfig, self).__init__(
            description=description,
            version=datasets.Version("1.0.0"),
            **kwargs,
        )
        self.language_pair = language_pair


class MTVietnameseBahnaric(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ViBaDatasetConfig(
            name="ViBaDataset",
            language_pair=("vi", "ba"),
        )
    ]
    BUILDER_CONFIG_CLASS = ViBaDatasetConfig

    def _info(self):
        source, target = self.config.language_pair
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"translation": datasets.features.Translation(languages=self.config.language_pair)}
            ),
            supervised_keys=(source, target)
        )

    def _split_generators(self, dl_manager):
        source, target = self.config.language_pair

        files = {}
        for split in ("train", "valid", "test"):
            if split == "train":
                dl_dir = "viba_dataset/train.json"
            if split == "valid":
                dl_dir = "viba_dataset/valid.json"
            if split == "test":
                dl_dir = "viba_dataset/test.json"

            files[split] = {"source_file": dl_dir}

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=files["train"]),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=files["valid"]),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=files["test"]),
        ]

    def _generate_examples(self, source_file):
        data = json.load(open(source_file, "r"))["data"]
        shuffle(data)
        # if "valid" in source_file:
        #     data = data[:40]
        # elif "test" in source_file:
        #     data = data[:1000]
        truth, typo = self.config.language_pair
        for idx, data_item in enumerate(data):
            result = {"translation": data_item}
            # Make sure that both translations are non-empty.
            yield idx, result