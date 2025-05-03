import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.rec_base_dataset_builder import RecBaseDatasetBuilder
# from minigpt4.datasets.datasets.laion_dataset import LaionDataset
# from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset

from minigpt4.datasets.datasets.rec_datasets import MoiveOOData, MoiveOOData_sasrec, AmazonOOData, AmazonOOData_sasrec



@registry.register_builder("movie_ood")
class MoiveOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
            seq_len=build_info.seq_len
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')],
            seq_len=build_info.seq_len
            )
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')],
            seq_len=build_info.seq_len
            )
            if evaluate_only:
                datasets['test_warm'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_warm_cold=warm')],
                seq_len=build_info.seq_len
                )

                datasets['test_cold'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_warm_cold=cold')],
                seq_len=build_info.seq_len
                )
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("movie_ood_sasrec")
class MoiveOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
            seq_len=build_info.seq_len
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')],
            seq_len=build_info.seq_len
            )
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')],
            seq_len=build_info.seq_len
            )
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets



@registry.register_builder("amazon_ood")
class AmazonOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self, evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        sas_seq_len = build_info.get("sas_seq_len", None)
        print(f'build_info:{build_info}')

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
            seq_len=build_info.seq_len,
            user2group=build_info.user2group,
            use_ids=build_info.use_ids,
            use_desc=build_info.use_desc,
            sas_seq_len=sas_seq_len
        )
        if not evaluate_only:
            try:
                datasets['reason'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'reason')],
                seq_len=build_info.seq_len,
                user2group=build_info.user2group,
                use_ids=build_info.use_ids,
                use_desc=build_info.use_desc,
                sas_seq_len=sas_seq_len
                )
            except:
                print("Not existing",os.path.join(storage_path, 'reason_ood2.pkl'))
            try:
                datasets['valid'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'valid_small')],
                seq_len=build_info.seq_len,
                user2group=build_info.user2group,
                use_ids=build_info.use_ids,
                use_desc=build_info.use_desc,
                sas_seq_len=sas_seq_len
                )
            except:
                print("Not existing",os.path.join(storage_path, 'valid_small'))
        else:
            try:
                #0915
                datasets['test'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test')],
                seq_len=build_info.seq_len,
                user2group=build_info.user2group,
                use_ids=build_info.use_ids,
                use_desc=build_info.use_desc,
                sas_seq_len=sas_seq_len
                )
            except Exception as e:
                print("Not existing",os.path.join(storage_path, 'test_ood2.pkl'),e)
            try:
                datasets['test_tiny'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_tiny')],
                seq_len=build_info.seq_len,
                user2group=build_info.user2group,
                use_ids=build_info.use_ids,
                use_desc=build_info.use_desc,
                sas_seq_len=sas_seq_len
                )
            except Exception as e:
                print("Not existing",os.path.join(storage_path, 'test_tiny_ood2.pkl'),e)
                # datasets['test_warm'] = dataset_cls(
                # text_processor=self.text_processors["train"],
                # ann_paths=[os.path.join(storage_path, 'test=warm')],
                # seq_len=build_info.seq_len,
                # user2group=build_info.user2group,
                # use_ids=build_info.use_ids,
                # use_desc=build_info.use_desc,
                # sas_seq_len=sas_seq_len
                # )
                # datasets['test_cold'] = dataset_cls(
                # text_processor=self.text_processors["train"],
                # ann_paths=[os.path.join(storage_path, 'test=cold')],
                # seq_len=build_info.seq_len,
                # user2group=build_info.user2group,
                # use_ids=build_info.use_ids,
                # use_desc=build_info.use_desc,
                # sas_seq_len=sas_seq_len
                # )
            try:
                datasets['test_small'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_small')],
                seq_len=build_info.seq_len,
                user2group=build_info.user2group,
                use_ids=build_info.use_ids,
                use_desc=build_info.use_desc,
                sas_seq_len=sas_seq_len
                )
            except:
                print("Not existing test_small, test loaded")
                datasets['test_small'] = datasets['test']
        return datasets


@registry.register_builder("amazon_ood_sasrec")
class AmazonOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
            seq_len=build_info.seq_len
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')],
            seq_len=build_info.seq_len
            )
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')],
            seq_len=build_info.seq_len
            )
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets
