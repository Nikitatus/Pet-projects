import torch
import torch.nn as nn
from typing import Tuple 
from collections import OrderedDict
import json

class LSTMCell(nn.Module):
    """A single LSTM cell implementation with explicit gates:
    input, forget, cell, and output gates.

    Args:
        input_size: Dimensionality of the input features.
        hidden_size: Dimensionality of the hidden and cell state.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        concatenated_size = hidden_size + input_size
        self.input_gate = nn.Linear(concatenated_size, hidden_size)
        self.forget_gate = nn.Linear(concatenated_size, hidden_size)
        self.cell_gate = nn.Linear(concatenated_size, hidden_size)
        self.output_gate = nn.Linear(concatenated_size, hidden_size)

    def forward(self, token: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_and_hidden = torch.cat([token, hidden_state], dim=-1)

        forget_activation = torch.sigmoid(self.forget_gate(token_and_hidden))
        cell_candidate = torch.tanh(self.cell_gate(token_and_hidden))
        input_activation = torch.sigmoid(self.input_gate(token_and_hidden))
        new_cell_state = forget_activation * cell_state + input_activation * cell_candidate

        output_activation = torch.sigmoid(self.output_gate(token_and_hidden))
        new_hidden_state = output_activation * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state

class LSTM(nn.Module):
    """A simple unidirectional LSTM implemented from scratch using LSTMCell.
    Processes sequences token by token to produce the final hidden state.

    Args:
        input_size: Dimensionality of the input features per token.
        hidden_size: Dimensionality of the hidden and cell states.
        batch_first: If True, input is shaped (batch, seq_len, input_size), (seq, batch, feature) – otherwise.
    """
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(self.input_size, self.hidden_size)
        self.batch_first = batch_first

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor :
        if self.batch_first:
            input_sequence = input_sequence.transpose(0, 1)
        batch_size = input_sequence.shape[1]
        hidden_state = input_sequence.new_zeros(batch_size, self.hidden_size)
        cell_state = input_sequence.new_zeros(batch_size, self.hidden_size)
        for token in input_sequence:
            hidden_state, cell_state = self.cell(token, hidden_state, cell_state)
        return hidden_state
    
class BidirectionalLSTM(nn.Module):
    """A custom implementation of a bidirectional LSTM using two unidirectional LSTMs:
    one processing the sequence in the forward direction and one in reverse.
    The outputs from both directions are concatenated to form the final hidden state.

    Args:
        input_size: Dimensionality of the input features per token.
        hidden_size: Number of hidden units in each LSTM direction.
        batch_first: If True, input/output tensors are shaped (batch, seq, feature), (seq, batch, feature) – otherwise.
    """
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool =True):
        super(BidirectionalLSTM, self).__init__()
        self.forward_network = LSTM(input_size, hidden_size, batch_first)
        self.backward_network = LSTM(input_size, hidden_size, batch_first)
        self.batch_first = batch_first

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        direct_sequence = input_sequence
        if self.batch_first:
            sequence_dim = 1
        else:
            sequence_dim = 0
        reverse_sequence = torch.flip(input_sequence, dims=[sequence_dim])

        forward_hidden_state = self.forward_network(direct_sequence)
        backward_hidden_state = self.backward_network(reverse_sequence)
        joint_hidden_state = torch.cat((forward_hidden_state, backward_hidden_state), dim=-1)
        return joint_hidden_state
    
def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

vocabulary_path = 'vocabulary.json'    
vocabulary = load_json(vocabulary_path)
vocabulary_size = len(vocabulary['id_to_token'])

embedding_size = 128
hidden_size = 64
num_classes = 3

classifier = nn.Sequential(OrderedDict(
    [
        ("embedding", nn.Embedding(vocabulary_size, embedding_size)),
        ("BiLSTM", BidirectionalLSTM(embedding_size, hidden_size)),
        ("output", nn.Linear(2 * hidden_size, num_classes))
    ]
))