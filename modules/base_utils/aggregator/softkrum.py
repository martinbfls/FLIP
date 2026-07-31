# coding: utf-8

import torch

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import combinations
from torch import Tensor
from typing import Any, Optional

# ---------------------------------------------------------------------------- #
# Combination helpers

def combination_indexes(n: int, p: int) -> Iterator[tuple[Iterable[int], Iterable[int]]]:
	selects = list(i < p for i in range(n))
	while True:
		# Emit current (anti-)selection
		yield (filter(lambda i: selects[i], range(n)), filter(lambda i: not selects[i], range(n)))
		# Advance to next selection (if any)
		cursor = len(selects) - 1
		target = None
		passed = 0
		while True:
			if cursor < 0:
				return
			elif selects[cursor]:
				if target is None:
					selects[cursor] = False
					passed += 1
				else:
					selects[cursor] = False
					selects[target] = True
					while passed > 0:
						target += 1
						passed -= 1
						selects[target] = True
					break
			else:
				target = cursor
			cursor -= 1

# ---------------------------------------------------------------------------- #
# Soft helpers

def broadshape(tensor: Tensor) -> Iterator[int]:
	shape = iter(tensor.shape)
	try:
		yield next(shape)
		while True:
			next(shape)
			yield 1
	except StopIteration:
		pass

def softmax(items: Tensor, negate: bool = False, sharpness: float = 1.) -> Tensor:
	if len(items) < 2:
		return torch.ones_like(items)
	items = items.div(items.std(dim=0))
	if negate:
		sharpness = -sharpness
	result = items.mul(sharpness).softmax(dim=0)
	return result

def softnth(items: Tensor, select: int, reduce: Callable[[tuple[int, ...], Tensor], Tensor], sharpness: float = 1.) -> Tensor:
	assert len(items.shape) == 1, "expected a list of scalars"
	reduceds = list()
	criterions = list()
	for sels, rejs in combination_indexes(len(items), select):
		indexes = tuple(sels)
		sels = torch.stack(tuple(items[i] for i in indexes))
		rejs = torch.stack(tuple(items[i] for i in rejs))
		smax = softmax(sels, negate=False, sharpness=sharpness).mul(sels).sum()
		rmin = softmax(rejs, negate=True, sharpness=sharpness).mul(rejs).sum()
		reduceds.append(reduce(indexes, sels))
		criterions.append(rmin.sub(smax))
	reduceds = torch.stack(reduceds)
	criterions = torch.stack(criterions).view(*broadshape(reduceds))
	return softmax(criterions, sharpness=sharpness).mul(reduceds)

# ---------------------------------------------------------------------------- #
# Multi-Krum

def krum(*tensors: Tensor, f: int, m: int = 1, sharpness: float = 1.) -> Tensor:
	n = len(tensors)
	# Compute all pairwise distances
	distances = list()
	for x, y in combinations(tensors, r=2):
		distances.append(torch.dist(x, y, p=2))
	# Compute the scores
	scores = list()
	for i in range(len(tensors)):
		# Collect the distances
		distances_for_i = list()
		for j in range(i):
			distances_for_i.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
		for j in range(i + 1, n):
			distances_for_i.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
		# Select the `n - f - 2` smallest distances
		score = softnth(torch.stack(distances_for_i), n - f - 2, lambda _, distances: distances.sum(), sharpness=sharpness).sum(dim=0)
		scores.append(score)
	# Select the `m` lowest-scoring gradients
	scores = torch.stack(scores)
	return softnth(scores, m, lambda indexes, _: torch.stack(tuple(tensors[i] for i in indexes)).mean(dim=0), sharpness=sharpness).sum(dim=0)

# ---------------------------------------------------------------------------- #
# Playground

def truth_krum(*tensors: Tensor, f: int, m: int = 1) -> Tensor:
	n = len(tensors)
	# Compute all pairwise distances
	distances = list()
	for x, y in combinations(tensors, r=2):
		distances.append(torch.dist(x, y, p=2))
	# Compute the scores
	scores = list()
	for i in range(len(tensors)):
		# Collect the distances
		distances_for_i = list()
		for j in range(i):
			distances_for_i.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
		for j in range(i + 1, n):
			distances_for_i.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
		# Select the n - f - 2 smallest distances
		distances_for_i.sort()
		score = sum(distances_for_i[:n - f - 2])
		scores.append(score)
	# Select the `m` lowest-scoring gradients
	return sum(grad for _, grad in sorted((score, tensors[index]) for index, score in enumerate(scores))[:m]) / m

if __name__ == "__main__":

	for i in range(10):
		if i > 0:
			print()
		gradients = tuple(torch.rand(3) for _ in range(5))
		aggregated = krum(*gradients, f=1, m=2, sharpness=10.)
		truth = truth_krum(*gradients, f=1, m=2)
		print("multi-krum(")
		remain = len(gradients)
		for gradient in gradients:
			remain -= 1
			tail = "," if remain > 0 else ")"
			print(f"\t{gradient.tolist()}{tail}")
		print(f"= {aggregated.tolist()} (soft)")
		print(f"= {truth.tolist()} (truth)")
