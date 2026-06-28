#!/usr/bin/env python3
import warnings

warnings.filterwarnings("ignore", message=".*NotOpenSSL.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

from voice_brain_module.main import main
main()
