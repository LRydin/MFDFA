import sys
sys.path.append("../")

from MFDFA import singspect
from MFDFA import emddetrender


def test_exceptions():
    try:
        singspect._missing_library()
    except Exception:
        pass

    try:
        emddetrender._missing_library()
    except Exception:
        pass
