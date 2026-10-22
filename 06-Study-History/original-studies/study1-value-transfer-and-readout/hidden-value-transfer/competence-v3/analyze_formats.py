"""Use unchanged development analysis on the calibrated-format output folder."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'competence-v2'))
import analyze
analyze.HERE=HERE
if __name__=='__main__':analyze.main()
