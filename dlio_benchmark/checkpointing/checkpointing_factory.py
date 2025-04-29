import logging
from dlio_benchmark.common.enumerations import CheckpointMechanismType
from dlio_benchmark.common.error_code   import ErrorCodes
from dlio_benchmark.utils.config        import ConfigArguments, utcnow
from dlio_benchmark.utils.utility       import DLIOMPI

from dlio_benchmark.checkpointing.tf_checkpointing      import TFCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.checkpointing.s3_checkpoint_writer  import S3CheckpointWriter


_MECH_MAP = {
    CheckpointMechanismType.TF_SAVE : TFCheckpointing,
    CheckpointMechanismType.PT_SAVE : PyTorchCheckpointing,
    CheckpointMechanismType.CUSTOM  : S3CheckpointWriter,   # NEW
}


class CheckpointingFactory:
    """
    Return a singleton checkpointing implementation.
    """

    @staticmethod
    def get_mechanism(mech_type: CheckpointMechanismType):
        cfg = ConfigArguments.get_instance()

        # user can override via YAML
        if cfg.checkpoint_mechanism_class is not None:
            if DLIOMPI.get_instance().rank() == 0:
                logging.info(f"{utcnow()} Using custom checkpointing class "
                             f"{cfg.checkpoint_mechanism_class.__name__}")
            return cfg.checkpoint_mechanism_class.get_instance()

        if mech_type in _MECH_MAP:
            return _MECH_MAP[mech_type].get_instance()

        raise Exception(str(ErrorCodes.EC1005))

