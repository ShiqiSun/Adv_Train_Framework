
import utils.attack.attack_registers
from utils.register.registers import ATTACKS


def build_attacker(atk_type, log=None):

    attacker = ATTACKS[atk_type]
    log.logger.info(f"Attacker:{atk_type}.")

    return attacker

