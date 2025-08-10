from typing import Iterator, Optional, Self

from .graph import Graph
from .permutation import Permutation


class Glink:
    """
    A class representing a glink. 
    """
    __perm: Permutation
    __graph: Graph
    __next: Self

    def __init__(
        self,
        graph: Graph,
        perm: Permutation,
        next: Optional[Self] = None
    ):
        self.__graph = graph
        self.__perm = perm
        self.__next = next

    @property
    def graph(self) -> Optional[Graph]:
        return self.__graph

    @property
    def perm(self) -> Permutation:
        return self.__perm
    
    @property
    def next(self) -> Optional[Self]:
        return self.__next
    
    def link_to(self, glink: Self):
        self.__next = glink


class GlinkChain:
    """
    A linkedlist-like class representing a glink chain.
    """
    __head: Glink
    __tail: Glink
    __size: int

    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__size = 0

    def __len__(self) -> int:
        return self.__size

    def append(self, graph: Graph, perm: Permutation):
        """
        Add a new glink to the end of the chain.

        Params:
        - `graph`: the graph of the new glink
        - `perm`: the permutation inducing the glink
        """
        glink = Glink(graph, perm)
        if self.__head is None:
            self.__head = glink
        else:
            self.__tail.link_to(glink)
        self.__tail = glink
        self.__size += 1

    @property
    def head(self) -> Glink:
        return self.__head
    
    @property
    def tail(self) -> Glink:
        return self.__tail
    
    def glinks(self) -> Iterator[Glink]:
        """
        Iterate through each glink in the chain.

        Returns:
        - an iterator over the glinks.
        """
        glink = self.head
        while glink is not None:
            yield glink
            glink = glink.next
