import esm
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from scipy.spatial.distance import squareform, pdist, cdist
import string
from Bio import SeqIO

def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


def main():
    PDB_IDS = ["1a3a", "5ahw", "1xcr"]
    msas = {
        name: read_msa(f"D:\Python\esm\examples\data\{name.lower()}_1_A.a3m")
        for name in PDB_IDS
    }
    sequences = {
        name: msa[0] for name, msa in msas.items()
    }
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
    msa_transformer_predictions = {}
    msa_transformer_results = []
    print("-----------------modelandalphabet---------------")
    print(msa_transformer)
    print(msa_transformer_alphabet)
    for name, inputs in msas.items():
        inputs = greedy_select(inputs, num_seqs=128)  # can change this to pass more/fewer sequences
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter(
            [inputs])
        with torch.no_grad():
            results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=True)
        token_representations = results["representations"][12]
        print(token_representations.shape)
        token_representations = torch.squeeze(token_representations, 0)
        mean = torch.mean(token_representations, dim=0)
        print(mean.shape)
        token_representations = token_representations[0, 1:, ::]
        # logits = results["logits"]
        # contact = results["contacts"]
        # print(contact.shape)
        # print(logits.shape)
        print(msa_transformer_batch_tokens.shape)
        print(token_representations.shape)
if __name__ == "__main__":
    main()