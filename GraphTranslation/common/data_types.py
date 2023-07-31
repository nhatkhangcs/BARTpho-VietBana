from enum import Enum


class RelationTypes:
    NEXT = "NEXT"
    TRANSLATE = "TRANSLATE"
    SYNONYM = "SYNONYM"
    MAPPING = "MAPPING"
    CO_OCCURRENCE = "CO_OCCURRENCE"


class NodeType(Enum):
    GRAPH_WORD = 0
    SENT_SYLLABLE = 1
    SENT_WORD = 2

