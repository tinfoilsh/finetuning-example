#!/bin/bash
# Seeds the encrypted workspace on first start, then runs JupyterLab on :8888.
# The login token comes from the JUPYTER_TOKEN secret declared in tinfoil-config.yml.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
mkdir -p "$WORKSPACE/data" "$WORKSPACE/adapters" \
         "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"

# Never overwrite a notebook or data the user has edited; the workspace persists.
cp -n /opt/example/finetune.ipynb "$WORKSPACE/finetune.ipynb" || true
cp -n /opt/example/data/train.jsonl "$WORKSPACE/data/train.jsonl" || true
cp -n /opt/example/data/eval.jsonl "$WORKSPACE/data/eval.jsonl" || true

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  echo "JUPYTER_TOKEN is not set; refusing to start an unauthenticated notebook" >&2
  exit 1
fi

# The enclave's shim terminates TLS and forwards to this port with Host rewritten,
# so the browser's origin never matches the Host header: allow any origin and
# rely on the token for access control.
exec jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.root_dir="$WORKSPACE" \
  --ServerApp.default_url="/lab/tree/finetune.ipynb" \
  --ServerApp.allow_remote_access=True \
  --ServerApp.allow_origin='*' \
  --ServerApp.trust_xheaders=True \
  --ServerApp.terminals_enabled=False \
  --IdentityProvider.token="$JUPYTER_TOKEN" \
  --ServerApp.password='' \
  --LabApp.check_for_updates_class='jupyterlab.NeverCheckForUpdate'
