import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg", force=True)

from credit_sim.analysis import main  # noqa: E402

if __name__ == "__main__":
    main()
