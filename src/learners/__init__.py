from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .HPQ_learner import HPQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .HGCN_learner import HGCNLeaner
from .shaq_learner import SHAQLearner
from .demix_learner import DEMLearner
from .qnam_learner import QNAM_Learner
from .vaos_learner import VAOSLearner
from .polyline_learner import PolylineLearner
from .HIVE_learner import HGCNLeaner_new

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["HPQ_learner"] = HPQLearner
REGISTRY["hgcn_learner"] = HGCNLeaner
REGISTRY["shaq_learner"] = SHAQLearner
REGISTRY["demix_learner"] = DEMLearner
REGISTRY["qnam_learner"] = QNAM_Learner
REGISTRY["vaos_learner"] = VAOSLearner
REGISTRY["polyline_learner"] = PolylineLearner
REGISTRY["HIVE_learner"] = HGCNLeaner_new
