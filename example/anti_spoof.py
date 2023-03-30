"""License for the Code.

Copyright Owner: Yoshitsugu Kesamaru
Please refer to the separate license file for the license of the code.
"""

# Operate directory: Common to all examples
import os.path
import sys
dir: str = os.path.dirname(__file__)
parent_dir, _ = os.path.split(dir)
sys.path.append(parent_dir)



from face01lib.spoof import Spoof

# Spoof().iris()
# Spoof().obj_detect()
Spoof().make_qr_code()