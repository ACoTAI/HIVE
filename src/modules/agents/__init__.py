from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .ernn_agent import ERNNAgent
from .rnn_sd_agent import RNN_SD_Agent
from .central_rnn_agent import CentralRNNAgent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["ernn"] = ERNNAgent
REGISTRY["rnn_sd"] = RNN_SD_Agent
REGISTRY["central_rnn"] = CentralRNNAgent