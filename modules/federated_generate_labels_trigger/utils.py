from modules.base_utils.datasets import MTTDataset


class TriggerMTTDataset(MTTDataset):
    '''
    Thin wrapper around `MTTDataset` (modules/base_utils/datasets.py) that additionally
    returns a boolean `is_poisoned` flag per example: True iff this draw's "train" branch
    came from the appended poison segment (original i >= len(self.distill)), i.e. iff it is
    one of the genuinely-triggered-and-relabeled examples, as opposed to a plain pass-through
    clean example that happens to share the same ConcatDataset.

    This is needed because MTTDataset's returned `idx` (the 5th tuple element) alone cannot
    distinguish the two cases: for poisoned draws, idx is reassigned to
    `poison_inds[i % len(distill)]`, but a *non*-reassigned i (from the clean segment) can
    coincidentally also be a member of poison_inds (that source-class image, drawn
    unpoisoned) -- checking `idx in poison_inds` from outside would give false positives.

    Constructed by re-wrapping an existing MTTDataset's constituent objects (train, distill,
    poison_inds, transform, n_classes) -- e.g. the one returned by
    `modules.base_utils.datasets.get_matching_datasets` -- rather than duplicating its
    dataset-construction logic.
    '''

    def __getitem__(self, i: int):
        is_poisoned = i >= len(self.distill)
        train_x, train_oh, distill_x, distill_oh, idx = super().__getitem__(i)
        return train_x, train_oh, distill_x, distill_oh, idx, is_poisoned

    @classmethod
    def from_mtt_dataset(cls, mtt_dataset):
        return cls(
            mtt_dataset.train, mtt_dataset.distill, mtt_dataset.poison_inds,
            mtt_dataset.transform, mtt_dataset.n_classes,
        )
