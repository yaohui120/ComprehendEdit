from ..trainer import MEND, MEND_multigpu
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import MALMEN
from ..trainer import FT, HICE


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'MALMEN': MALMEN,
    'MEND_MULTIGPU':MEND_multigpu,
    'HICE': HICE,
    'FT': FT,
}