from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from megatron import mpu
from megatron.data import data_utils, indexed_dataset


OUTPUT_DIR = Path(__file__).resolve().parent
PREFIX = OUTPUT_DIR / "dummy_text_document"
SEQ_LENGTH = 4


def build_dummy_corpus():
    documents = [
        np.arange(101, 106, dtype=np.int32),
        np.arange(201, 208, dtype=np.int32),
        np.arange(301, 305, dtype=np.int32),
    ]
    builder = indexed_dataset.make_builder(str(PREFIX) + ".bin", "mmap")
    for document in documents:
        builder.add_item(document)
        builder.end_document()
    builder.finalize(str(PREFIX) + ".idx")
    return documents


def main():
    documents = build_dummy_corpus()
    rendezvous = OUTPUT_DIR / "distributed_init"
    rendezvous.unlink(missing_ok=True)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    mpu.initialize_model_parallel(1)

    try:
        with patch.object(
            torch.cuda,
            "LongTensor",
            side_effect=lambda values: torch.tensor(values, dtype=torch.long),
        ):
            common_args = dict(
                data_prefix=str(PREFIX),
                pos_data_prefix=None,
                neg_data_prefix=None,
                data_impl="mmap",
                pack_impl="packed",
                dataset_impl="gpt2",
                allow_chopped=True,
                num_samples=6,
                num_epochs=None,
                seq_length=SEQ_LENGTH,
                seed=123,
                skip_warmup=True,
            )
            ordered_dataset = data_utils.build_the_dataset(
                **common_args,
                name="train",
                preserve_data_order=True,
            )
            # Deliberately omit preserve_data_order to exercise its default value.
            shuffled_dataset = data_utils.build_the_dataset(
                **common_args,
                name="train_default",
            )

        expected_doc_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        expected_shuffle_idx = np.arange(len(ordered_dataset.shuffle_idx))
        np.testing.assert_array_equal(ordered_dataset.doc_idx, expected_doc_idx)
        np.testing.assert_array_equal(
            ordered_dataset.shuffle_idx, expected_shuffle_idx
        )
        assert not np.array_equal(shuffled_dataset.doc_idx, expected_doc_idx)
        assert not np.array_equal(
            shuffled_dataset.shuffle_idx,
            np.arange(len(shuffled_dataset.shuffle_idx)),
        )

        expected_stream = np.concatenate(documents * 2)
        ordered_samples = [
            ordered_dataset[i]["text"] for i in range(len(ordered_dataset))
        ]
        shuffled_samples = [
            shuffled_dataset[i]["text"] for i in range(len(shuffled_dataset))
        ]
        for index, sample in enumerate(ordered_samples):
            expected = expected_stream[
                SEQ_LENGTH * index : SEQ_LENGTH * index + SEQ_LENGTH + 1
            ]
            np.testing.assert_array_equal(sample, expected)
        assert any(
            not np.array_equal(ordered, shuffled)
            for ordered, shuffled in zip(ordered_samples, shuffled_samples)
        )

        print(f"prefix={PREFIX}")
        print(f"document_sizes={[len(document) for document in documents]}")
        print(f"ordered_doc_idx={ordered_dataset.doc_idx.tolist()}")
        print(f"ordered_shuffle_idx={ordered_dataset.shuffle_idx.tolist()}")
        print(
            f"ordered_samples={[sample.tolist() for sample in ordered_samples]}"
        )
        print(f"default_doc_idx={shuffled_dataset.doc_idx.tolist()}")
        print(f"default_shuffle_idx={shuffled_dataset.shuffle_idx.tolist()}")
        print(
            f"default_samples={[sample.tolist() for sample in shuffled_samples]}"
        )
        print("PASS: flag preserves order and the default path shuffles it")
    finally:
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
