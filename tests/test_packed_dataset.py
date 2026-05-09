import torch

from valkmodel.data import PackedDataset, create_document_attention_mask, pack_documents


def test_pack_documents_inserts_separators_pads_and_ignores_padding_labels():
    packed = pack_documents(
        documents=[[5, 6], [7], [8, 9, 10]],
        max_seq_len=8,
        pad_token_id=0,
        separator_token_id=99,
    )[0]

    assert packed["input_ids"].tolist() == [5, 6, 99, 7, 99, 8, 9, 10]
    assert packed["labels"].tolist() == [5, 6, 99, 7, 99, 8, 9, 10]
    assert packed["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert packed["document_ids"].tolist() == [0, 0, -1, 1, -1, 2, 2, 2]
    assert packed["document_boundaries"] == [(0, 2), (3, 4), (5, 8)]


def test_pack_documents_chunks_oversized_documents_and_pads_tail():
    packed = pack_documents(
        documents=[[1, 2, 3, 4, 5, 6]],
        max_seq_len=4,
        pad_token_id=0,
    )

    assert len(packed) == 2
    assert packed[0]["input_ids"].tolist() == [1, 2, 3, 4]
    assert packed[1]["input_ids"].tolist() == [5, 6, 0, 0]
    assert packed[1]["labels"].tolist() == [5, 6, -100, -100]
    assert packed[1]["attention_mask"].tolist() == [1, 1, 0, 0]


def test_packed_dataset_returns_tensors_with_boundary_metadata():
    dataset = PackedDataset(
        documents=[[1, 2], [3, 4, 5]],
        max_seq_len=6,
        pad_token_id=0,
        separator_token_id=99,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].tolist() == [1, 2, 99, 3, 4, 5]
    assert item["document_boundaries"] == [(0, 2), (3, 6)]


def test_document_attention_mask_blocks_padding_and_cross_document_attention():
    document_ids = torch.tensor([[0, 0, -1, 1, 1, -1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])

    mask = create_document_attention_mask(document_ids, attention_mask)

    assert mask.shape == (1, 6, 6)
    assert mask[0, 1, 0]
    assert not mask[0, 3, 1]
    assert mask[0, 4, 3]
    assert not mask[0, 5].any()
