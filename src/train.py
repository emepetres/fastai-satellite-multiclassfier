import os
from pathlib import Path
from learner import get_learner

if __name__ == "__main__":
    os.chdir(Path(os.path.abspath(os.path.dirname(__file__))))

    learn = get_learner(bs=64)

    learn.fit_one_cycle(1, slice(2e-3))

    learn.unfreeze()
    learn.fit_one_cycle(5, slice(2e-3 / 2.6**4, 2e-3))
    # This lr/2.6**4 is a general rule of thumb when doing gradual unfreezing

    learn.save("weights")
