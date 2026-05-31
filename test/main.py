#!/usr/bin/env python3
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

warnings.filterwarnings("ignore", message=".*NotOpenSSL.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

from realtime_asr.main import main
main()
