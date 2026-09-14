"""Generates notebook/finetune.ipynb. Run after editing: python notebook/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS = [
    ("md", """# Fine-tune a model privately, inside an enclave

This notebook is running inside a [Tinfoil Container](https://docs.tinfoil.sh/containers/overview): a hardware-attested enclave with one GPU. Three things are true here that are not true on an ordinary GPU box:

- **The base model is verified.** `google/gemma-4-E2B-it` is mounted read-only from a model pack whose root hash is part of the enclave measurement.
- **Your data and your adapter stay private.** `/workspace` is an encrypted, integrity-protected disk. It was unlocked at boot with a key released only to this measured enclave, and it survives restarts and updates.
- **Nothing leaves.** The enclave has no network egress. The only way in or out is this notebook, over the attested TLS connection you are using right now.

Run the cells top to bottom (**Run ▸ Run All Cells**). End to end takes about three minutes on one GPU."""),

    ("code", '''import json, math, os, random, time
from pathlib import Path

import torch, yaml

MODEL_DIR = os.environ["MODEL_DIR"]
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))

config = yaml.safe_load(Path("/tinfoil/config.yml").read_text())
volume = config["volumes"][0]
mount = next(line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines() if line.split()[4] == str(WORKSPACE))

print("cvm-version :", config["cvm-version"])
print("base model  :", config["models"][0]["repo"])
print("workspace   :", f"volume '{volume['name']}' unlocked at boot with secret {volume['key-secret']}")
print("mounted from:", mount[-2], f"({mount[-3]})")
print("egress      :", "none (no `networks:` in the measured config)" if not config.get("networks") else config["networks"])
print("GPU         :", torch.cuda.get_device_name(0))'''),

    ("md", """## 1. Load the verified base model

The weights come from `/tinfoil/mpk/...`, a read-only mount of the model pack pinned in `tinfoil-config.yml`. Nothing is downloaded: the enclave could not reach Hugging Face even if it wanted to."""),

    ("code", '''from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
print(f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters loaded on {model.device}")


def ask(question, max_new_tokens=96):
    """Greedy answer to a single user turn."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()'''),

    ("md", """## 2. Your private data

Training data is a JSONL file of chat conversations, one per line: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. Drop your own file into `data/` using the file browser on the left; it lands on the encrypted volume and never touches the host.

The sample, `data/train.jsonl`, is an internal helpdesk for a fictional coffee company. Every fact in it is invented, so the base model cannot know any of them. That makes the before/after obvious."""),

    ("code", '''def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

train_rows = read_jsonl(WORKSPACE / "data" / "train.jsonl")
eval_rows = read_jsonl(WORKSPACE / "data" / "eval.jsonl")
print(f"{len(train_rows)} training conversations, {len(eval_rows)} held-out questions\\n")
for row in train_rows[:2]:
    print("user     :", row["messages"][0]["content"])
    print("assistant:", row["messages"][1]["content"].replace("\\n", " "), "\\n")'''),

    ("md", """### Before training

Ask the base model three of the held-out questions. It has never seen this company, so it guesses."""),

    ("code", '''EVAL_QUESTIONS = [row["messages"][0]["content"] for row in eval_rows[:3]]
for question in EVAL_QUESTIONS:
    print(f"Q: {question}\\nA: {ask(question)}\\n")'''),

    ("md", """## 3. Tokenize

Each conversation is rendered with the model's chat template. Loss is computed on the assistant's turn only: the prompt tokens get the label `-100`, which PyTorch ignores."""),

    ("code", '''def encode(row):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt), "chat template must render the prompt as a prefix of the conversation"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    return {"input_ids": ids, "labels": labels}

encoded = [encode(row) for row in train_rows]
lengths = [len(e["input_ids"]) for e in encoded]
print(f"{len(encoded)} examples, {min(lengths)}-{max(lengths)} tokens each")


def collate(batch):
    width = max(len(e["input_ids"]) for e in batch)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([e["input_ids"] + [pad] * (width - len(e["input_ids"])) for e in batch])
    labels = torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in batch])
    attention_mask = torch.tensor([[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}'''),

    ("md", """## 4. Attach LoRA adapters

LoRA trains a few million extra parameters on top of the frozen base model. The adapters attach to the attention and MLP projections of the language model only (Gemma 4 also carries vision and audio towers, which stay untouched)."""),

    ("code", '''from peft import LoraConfig, get_peft_model

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=r"model\\.language_model\\.layers\\.\\d+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))",
)
model = get_peft_model(model, lora)
for parameter in model.parameters():
    if parameter.requires_grad:
        parameter.data = parameter.data.float()  # keep the trainable weights in fp32; the base stays bf16
model.print_trainable_parameters()'''),

    ("md", """## 5. Train

A plain PyTorch loop, so there is nothing hidden: forward, backward, clip, step. The loss curve updates live."""),

    ("code", '''import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from transformers import get_linear_schedule_with_warmup

EPOCHS, BATCH_SIZE, LEARNING_RATE = 3, 8, 2e-4

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
steps_per_epoch = math.ceil(len(encoded) / BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

random.seed(0)
model.train()
losses, step, started = [], 0, time.time()
figure, axis = plt.subplots(figsize=(7, 3))
for epoch in range(EPOCHS):
    order = random.sample(range(len(encoded)), len(encoded))
    for start in range(0, len(order), BATCH_SIZE):
        batch = collate([encoded[i] for i in order[start:start + BATCH_SIZE]])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        axis.clear()
        axis.plot(losses)
        axis.set(xlabel="step", ylabel="loss", title=f"epoch {epoch + 1}/{EPOCHS}   step {step}/{total_steps}   loss {losses[-1]:.3f}")
        clear_output(wait=True)
        display(figure)
plt.close(figure)
model.eval()
print(f"{step} steps in {time.time() - started:.0f}s, loss {losses[0]:.2f} -> {losses[-1]:.3f}")'''),

    ("md", """## 6. Save the adapter to the encrypted workspace

`save_pretrained` writes the LoRA weights (a few tens of megabytes) into `adapters/<run>/` on the encrypted volume. They stay there across container restarts and updates, and the host never sees them in the clear."""),

    ("code", '''run_dir = WORKSPACE / "adapters" / time.strftime("%Y%m%d-%H%M%S")
model.save_pretrained(run_dir)
(run_dir / "training.json").write_text(json.dumps({
    "base_model": config["models"][0]["repo"],
    "examples": len(encoded), "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
    "final_loss": losses[-1], "losses": losses,
}, indent=2))
for path in sorted(run_dir.iterdir()):
    print(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(WORKSPACE)}")'''),

    ("md", """## 7. Before and after

Same questions, same weights, adapter off then on. Everything the tuned model knows about the company came from `data/train.jsonl`."""),

    ("code", '''for question in EVAL_QUESTIONS:
    with model.disable_adapter():
        before = ask(question)
    after = ask(question)
    print(f"Q: {question}\\n   base : {before}\\n   tuned: {after}\\n")'''),

    ("md", """## 8. It persists

Stop and start the container from your laptop, then come back and run the next cell in a fresh kernel. The adapter is still on the encrypted volume, and it loads onto the freshly verified base model.

```bash
tinfoil container stop finetune && tinfoil container start finetune
```"""),

    ("code", '''import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
runs = sorted(p for p in (WORKSPACE / "adapters").iterdir() if (p / "adapter_config.json").exists())
print("adapters on the encrypted volume:", *[run.name for run in runs], sep="\\n  ")

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
base = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_DIR"], dtype=torch.bfloat16, device_map="cuda")
tuned = PeftModel.from_pretrained(base, runs[-1]).eval()

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "When does the Huila Reserve ship?"}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(tuned.device)
with torch.no_grad():
    out = tuned.generate(**inputs, max_new_tokens=64, do_sample=False)
print("\\n" + tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())'''),

    ("md", """## Next steps

- **Serve it.** Point a vLLM container at the adapter directory with `--enable-lora`; the [LoRA demo](https://github.com/tinfoilsh/confidential-lora-demo) shows a serving config.
- **Keep the key away from Tinfoil.** Set `keyserver-url` in `tinfoil-config.yml` and release `WORKSPACE_KEY` and `JUPYTER_TOKEN` from your own [keyserver](https://docs.tinfoil.sh/containers/private-secrets); the volume is then unreadable to the operator as well.
- **Bigger models.** Wrap `google/gemma-4-12B-it` or `google/gemma-4-31B-it` with `tinfoil model wrap`, paste the printed `models:` block, and release a new version. Nothing in this notebook changes.
- **Verify from outside.** `tinfoil attestation verify -e <your-domain> -r <owner>/<repo>` checks that the enclave you are talking to runs exactly the measured release."""),
]


def build():
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body) for kind, body in CELLS]
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbf.validate(notebook)
    nbf.write(notebook, HERE / "finetune.ipynb")
    print(f"wrote {HERE / 'finetune.ipynb'} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
