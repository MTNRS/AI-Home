"""Caption SFT examples with image-aware context and assistant-only targets."""

import torch
from torch.utils.data import Dataset

from .vision_data import load_visual_manifest, load_visual_record


class VisualCaptionDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, text_config, vision_config, *, split,
                 prompt="Describe the image."):
        if split not in ("train", "validation"):
            raise ValueError("Invalid visual dataset split")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise ValueError("Invalid caption instruction")
        if tokenizer.vocab_size != text_config.vocab_size:
            raise ValueError("Visual tokenizer vocabulary mismatch")
        records, self.dataset_sha256 = load_visual_manifest(manifest_path)
        self.manifest_path = manifest_path
        self.records = [r for r in records if r["split"] == split]
        self.image_size = vision_config.image_size
        self.context = text_config.context - vision_config.patches
        self.prompt = prompt
        self.sequences = []
        prefix = tokenizer.conversation([{"role": "user", "content": prompt}])
        for record in self.records:
            answer = [*tokenizer.encode(record["caption"]), 258]
            ids = prefix + answer
            # Labels are shifted once, so the assistant marker predicts caption[0].
            labels = [-100] * len(prefix) + answer
            inputs, targets = ids[:-1], labels[1:]
            if len(inputs) > self.context:
                raise ValueError("Visual caption exceeds context; enlarge context or review shorter text")
            padding = self.context - len(inputs)
            self.sequences.append((torch.tensor(inputs + [256] * padding, dtype=torch.long),
                                   torch.tensor(targets + [-100] * padding, dtype=torch.long)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, _, _ = load_visual_record(self.manifest_path, self.records[index], self.image_size)
        inputs, targets = self.sequences[index]
        return image.squeeze(0), inputs.clone(), targets.clone()
