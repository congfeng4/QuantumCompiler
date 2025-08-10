from typing import Iterator, Optional, Self


class Node:
    """
    Wrapper class representing a node.
    """
    def __init__(self, val: Optional[int]):
        if isinstance(val, Node):
            raise ValueError("data is already of type Node")
        self.__val = val

    @classmethod
    def null(cls) -> Self:
        return cls(None)
    
    @property
    def is_null(self) -> bool:
        return self.__val is None

    @property
    def val(self) -> int:
        return self.__val

    def __repr__(self) -> str:
        return str(self.__val) if not self.is_null else "NULL-NODE"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        if self.is_null:
            return False
        if self.__val != other.__val:
            return False
        return True
    
    def __gt__(self, other: Self) -> bool:
        return self.__val > other.__val
    
    def __lt__(self, other: Self) -> bool:
        return self.__val < other.__val
    
    def __hash__(self) -> int:
        val = self.__val if not self.is_null else -1
        return hash(val)
    

class Edge:
    """
    Wrapper class representing an undirected edge.
    """
    def __init__(self, src: Node, dst: Node):
        if isinstance(src, int) or isinstance(dst, int):
            raise ValueError("src and dst needs to be of type Node")
        self.__val = (src, dst)

    @classmethod
    def null(cls):
        return cls(Node.null(), Node.null())

    @property
    def is_null(self) -> bool:
        return self.__val[0].is_null and self.__val[1].is_null

    @property
    def val(self) -> tuple[Node, Node]:
        return self.__val

    def __repr__(self) -> str:
        src, dst = self.__val
        return f"{src}-{dst}" if not self.is_null else "NULL-EDGE"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Edge):
            return False
        if self.is_null:
            return False
        if self.__val not in (other.__val, other.__val[::-1]):
            return False
        return True
    
    def __hash__(self) -> int:
        val = self.__val[0].val + self.__val[1].val if not self.is_null else -1
        return hash(val)
    
    def __iter__(self) -> Iterator[Node]:
        return iter(self.__val)
