import random as rd

from typing import Optional, Self, Sequence

from .graph_data import Node, Edge


def formatted(src: Node, dst: Node):
    return str(dst) if dst != src else f"\033[2m{dst}\033[0m" # dimmed


class Permutation:
    """
    A dict-like object representing a permutation of nodes.
    """
    __perm: tuple[Edge]

    def __init__(self, *perm: Edge, type: str = "map"):
        if type not in ("map", "swap"):
            return ValueError(f"invalid type '{type}'")
        self.__type = type
        self.__perm = perm

    @classmethod
    def identity(cls) -> Self:
        return cls()

    @classmethod
    def random(cls, original: Sequence[Node]) -> Self:
        perm = zip(original, rd.sample(original, len(original)))
        return cls(*(Edge(src, dst) for src, dst in perm))

    def __len__(self) -> int:
        return len(self.__perm)
    
    def __repr__(self) -> str:
        return "Permutation(\n  {}\n)".format(
            "\n  ".join(str(edge) for edge in self.__perm)
        )
    
    def __getitem__(self, key: Node | int) -> Node:
        if isinstance(key, int):
            key = Node(key)
        return self.dict()[key]
    
    @property
    def type(self) -> str:
        return self.__type
    
    def keys(self) -> list[Node]:
        return [src for src, _ in self.items()]
    
    def values(self) -> list[Node]:
        return [dst for _, dst in self.items()]
    
    def items(self) -> tuple[Edge]:
        return self.__perm
    
    def __apply_map(
        self,
        original: Sequence[Node],
        inplace: bool = False
    ) -> Optional[list[Node]]:
        
        sigma = {src: dst for src, dst in self.items()}
        permuted = original if inplace else original.copy()
        
        for i, node in enumerate(original):
            if node not in sigma:
                continue
            permuted[i] = sigma[node]

        if not inplace:
            return permuted
    
    def __apply_swap(
        self,
        original: list[Node],
        inplace: bool = False
    ) -> Optional[list[Node]]:
        
        sigma = {node: original.index(node) for node in original}
        permuted = original if inplace else original.copy()

        for src, dst in self.items():
            permuted[sigma[src]], permuted[sigma[dst]] = \
                permuted[sigma[dst]], permuted[sigma[src]]
        
        if not inplace:
            return permuted
    
    def apply(
        self,
        original: list[Node],
        inplace: bool = False
    ) -> Optional[list[Node]]:
        """
        Apply the permutation on the original sequence.

        Params:
        - `original`: the original sequence to permute
        - `inplace`: if True then the permutation is performed in-place

        Returns:
        - the permuted sequence if not `inplace`.
        """
        if self.type == "map":
            return self.__apply_map(original, inplace)
        return self.__apply_swap(original, inplace)
    
    def oneline(
        self,
        original: Optional[list[Node]] = None,
        highlight: bool = True,
        padding: str = ""
    ) -> str:
        
        if original is None:
            original = self.keys()
        if highlight:
            permuted = (formatted(src, dst) for src, dst in zip(original, self.apply(original)))
        else:
            permuted = map(str, self.apply(original))
        return "{}({})".format(
            padding,
            " ".join(permuted)
        )
    
    def twoline(
        self,
        original: list[Node],
        highlight: bool = True,
        padding: str = ""
    ) -> str:
        
        if highlight:
            permuted = (formatted(src, dst) for src, dst in zip(original, self.apply(original)))
            original = (formatted(src, dst) for src, dst in zip(self.apply(original), original))
        else:
            permuted = map(str, self.apply(original))
            original = map(str, original)
        return "{0}({1})\n{0}({2})".format(
            padding,
            " ".join(original),
            " ".join(permuted)
        )
