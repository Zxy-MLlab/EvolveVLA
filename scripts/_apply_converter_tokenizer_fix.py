from pathlib import Path


path = Path("/root/autodl-tmp/EvolveVLA/scripts/robotwin_convert_exec.py")
text = path.read_text(encoding="utf-8")

replacements = [
    (
        "import argparse\n"
        "import hashlib\n"
        "import json\n"
        "import re\n"
        "import shutil\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from typing import Callable\n"
        "\n"
        "import numpy as np\n",
        "import argparse\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "import re\n"
        "import shutil\n"
        "import sys\n"
        "import traceback\n"
        "from pathlib import Path\n"
        "from typing import Callable\n"
        "\n"
        "os.environ.setdefault(\"HF_HUB_OFFLINE\", \"1\")\n"
        "os.environ.setdefault(\"TRANSFORMERS_OFFLINE\", \"1\")\n"
        "\n"
        "import numpy as np\n",
    ),
    (
        "    from lerobot.policies.pi05.configuration_pi05 import PI05Config\n"
        "    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors\n"
        "\n"
        "    config = PI05Config.from_pretrained(output, local_files_only=True)\n"
        "    if hasattr(config, \"tokenizer_processor\"):\n"
        "        config.tokenizer_processor.tokenizer_name = str(LOCAL_TOKENIZER)\n"
        "    tensor_stats = {\n"
        "        feature: {field: torch.tensor(values, dtype=torch.float32) for field, values in feature_stats.items()}\n"
        "        for feature, feature_stats in stats.items()\n"
        "    }\n"
        "    preprocessor, postprocessor = make_pi05_pre_post_processors(config, tensor_stats)\n",
        "    from lerobot.policies.pi05.configuration_pi05 import PI05Config\n"
        "    from lerobot.policies.factory import make_pre_post_processors\n"
        "\n"
        "    config = PI05Config.from_pretrained(output, local_files_only=True)\n"
        "    tensor_stats = {\n"
        "        feature: {field: torch.tensor(values, dtype=torch.float32) for field, values in feature_stats.items()}\n"
        "        for feature, feature_stats in stats.items()\n"
        "    }\n"
        "    preprocessor, postprocessor = make_pre_post_processors(\n"
        "        config,\n"
        "        preprocessor_overrides={\"tokenizer_processor\": {\"tokenizer_name\": str(LOCAL_TOKENIZER)}},\n"
        "        dataset_stats=tensor_stats,\n"
        "    )\n",
    ),
    (
        "    config = PI05Config.from_pretrained(output, local_files_only=True)\n"
        "    config.tokenizer_processor.tokenizer_name = str(LOCAL_TOKENIZER)\n"
        "    policy = PI05Policy(config)\n",
        "    config = PI05Config.from_pretrained(output, local_files_only=True)\n"
        "    if hasattr(config, \"tokenizer_processor\"):\n"
        "        config.tokenizer_processor.tokenizer_name = str(LOCAL_TOKENIZER)\n"
        "    policy = PI05Policy(config)\n",
    ),
    (
        "    except Exception as exc:\n"
        "        print(f\"PROCESSOR_WRITE_ERROR={type(exc).__name__}: {exc}\")\n",
        "    except Exception as exc:\n"
        "        traceback.print_exc()\n"
        "        print(f\"PROCESSOR_WRITE_ERROR={type(exc).__name__}: {exc}\")\n",
    ),
    (
        "    except Exception as exc:\n"
        "        print(f\"LEROBOT_LOAD_ERROR={type(exc).__name__}: {exc}\")\n"
        "        load_ok = False\n",
        "    except Exception as exc:\n"
        "        traceback.print_exc()\n"
        "        print(f\"LEROBOT_LOAD_ERROR={type(exc).__name__}: {exc}\")\n"
        "        load_ok = False\n",
    ),
    (
        "    except Exception as exc:\n"
        "        print(f\"CONVERT_ERROR={type(exc).__name__}: {exc}\")\n",
        "    except Exception as exc:\n"
        "        traceback.print_exc()\n"
        "        print(f\"CONVERT_ERROR={type(exc).__name__}: {exc}\")\n",
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:80]!r}")
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("patched", path)
