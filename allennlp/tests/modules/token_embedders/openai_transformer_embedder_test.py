# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
import numpy as np
import torch


# Skip this one, it's an expensive test.
from allennlp.modules.openai_transformer import OpenaiTransformer


@pytest.mark.skip()
class TestOpenaiTransformerEmbedderLarge(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'openai_transformer' / 'config_large.jsonnet',
                          self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}


class TestOpenaiTransformerEmbedderSmall(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'openai_transformer' / 'config_small.jsonnet',
                          self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}


@pytest.mark.skip()
class TestOpenaiTransformerReproducible(ModelTestCase):

    def setUp(self):
        super().setUp()
        transformer_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz"
        self.transformer_model = OpenaiTransformer(model_path=transformer_path, requires_grad=False)

        _input = np.load(
            self.FIXTURES_ROOT / 'openai_transformer' / 'openai_transformer_test_input.npy'
        )
        _output = np.load(
            self.FIXTURES_ROOT / 'openai_transformer' / 'openai_transformer_test_output.npy'
        )

        self.input_tensor = torch.LongTensor(_input)
        self.expected_output_tensor = torch.Tensor(_output)

    def test_embeddings(self):
        output = self.transformer_model(
            torch.LongTensor(self.input_tensor)
        )

        assert torch.all(torch.eq(output[-1], self.expected_output_tensor))
        assert not torch.all(torch.eq(output[0], self.expected_output_tensor))


def create_small_test_fixture(output_dir: str = '/tmp') -> None:
    """
    This is how I created the transformer_model.tar.gz.
    After running this, go to the specified output dir and run

        tar -czvf transformer_model.tar.gz model/

    In case you need to regenerate the fixture for some reason.
    """
    import json
    import pathlib
    from allennlp.modules.openai_transformer import OpenaiTransformer

    model_dir = pathlib.Path(output_dir) / 'model'
    model_dir.mkdir(exist_ok=True)  # pylint: disable=no-member

    symbols = ["e", "w", "o", "wo", "."]
    byte_pairs = [(sym1, sym2 + end)
                  for sym1 in symbols        # prefer earlier first symbol
                  for sym2 in symbols        # if tie, prefer earlier second symbol
                  for end in ('</w>', '')]   # if tie, prefer ending a word
    encoding = {f"{sym1}{sym2}": idx for idx, (sym1, sym2) in enumerate(byte_pairs)}
    encoding["<unk>"] = 0

    with open(model_dir / 'encoder_bpe.json', 'w') as encoder_file:
        json.dump(encoding, encoder_file)

    with open(model_dir / 'vocab.bpe', 'w') as bpe_file:
        bpe_file.write("#version 0.0\n")
        for sym1, sym2 in byte_pairs:
            bpe_file.write(f"{sym1} {sym2}\n")
        bpe_file.write("\n")

    transformer = OpenaiTransformer(embedding_dim=10, num_heads=2, num_layers=2, vocab_size=(50 + 50), n_ctx=50)
    transformer.dump_weights(output_dir, num_pieces=2)
