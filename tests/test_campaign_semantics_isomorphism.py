import networkx as nx
from topoformer.campaign_semantics_isomorphism import exact_equivalence


def test_exact_numbering_only_equivalence():
    a=nx.DiGraph();a.add_node(0,label=('pred',None,1));a.add_node(1,label=('ident',3,None));a.add_node(2,label=('ident',7,None))
    a.add_edge(0,1,label=((2,0),(4,0)));a.add_edge(0,2,label=((2,1),))
    b=nx.relabel_nodes(a,{0:12,1:8,2:4})
    assert exact_equivalence(a,b)[0]
    b.edges[12,8]['label']=((2,1),(4,1))
    assert not exact_equivalence(a,b)[0]
    b=nx.relabel_nodes(a,{0:12,1:8,2:4});b.nodes[8]['label']=('ident',9,None)
    assert not exact_equivalence(a,b)[0]
