# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging
import sys
from vllm.entrypoints.cli.main import main as vllm_main

logger = logging.getLogger(__name__)


def main() -> None:
    if "serve" in sys.argv and "--bs" in sys.argv:
        logger.warning(
            "vllm detected --bs flag in serve command. Launching beam-star optimization."
        )
        # Launch the specialized serving path
        from vllm.entrypoints.utils import cli_env_setup
        from beam_star.vllm_patch import patch_beam_search

        patch_beam_search()
        cli_env_setup()
        sys.argv.remove("--bs")

    # Delegate to original vLLM CLI
    vllm_main()


if __name__ == "__main__":
    main()
