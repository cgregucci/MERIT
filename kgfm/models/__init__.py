"""Re-exports the config-dispatchable model classes.

Configs reference these classes by short name (`Ultra`, `MOTIF`, `TRIXEntity`,
`TRIXRelation`, `Flock`, `MERIT`); entry-point scripts dispatch on
`cfg.model["class"]`.
"""

from kgfm.models.ultra import Ultra, RelNBFNet, EntityNBFNet
from kgfm.models.motif import MOTIF, RelHCNet
from kgfm.models.trix_entity import TRIXEntity
from kgfm.models.trix_relation import TRIXRelation
from kgfm.models.flock import Flock
from kgfm.models.merit import MERIT, RelTransformerMERIT, EntityNBFNetMERIT

__all__ = [
    "Ultra", "MOTIF", "TRIXEntity", "TRIXRelation", "Flock", "MERIT",
    "RelNBFNet", "EntityNBFNet", "RelHCNet",
    "RelTransformerMERIT", "EntityNBFNetMERIT",
]
