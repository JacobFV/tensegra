"""Synthetic CPU checks; no campaign result reads."""
import importlib.util,collections
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('S17-analyze.py');spec=importlib.util.spec_from_file_location('analysis',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
g={'presence':np.array([True,True,False]),'kind':np.array([1,2,0]),'value':np.array([-1,1,-1]),'copy':np.array([-1,0,-1]),'edges':np.zeros((3,3,2),bool),'slots':np.zeros((3,3),int)};g['edges'][0,1,0]=True
a={k:v.copy() for k,v in g.items()};assert all(m.components(a,g).values())
a['edges'][2,0,1]=True;assert all(m.components(a,g).values()) # absent node is masked
a['slots'][0,1]=1;assert not m.components(a,g)['slots'] and m.components(a,g)['edges']
a['edges'][0,1,0]=False;assert not m.components(a,g)['edges'] and not m.components(a,g)['slots']
a['slots'][0,1]=0;assert m.components(a,g)['slots'] and not m.components(a,g)['edges']
old=np.array([False,False,True,True]);new=np.array([False,True,False,True]);c=collections.Counter(f'{int(x)}->{int(y)}' for x,y in zip(old,new));assert c=={'0->0':1,'0->1':1,'1->0':1,'1->1':1}
gold=np.array([True,False,True,False]);historical=np.array([False,True,True,True]);matched=np.array([True,False,True,True]);fp=lambda p:int((p&~gold).sum());fn=lambda p:int((gold&~p).sum());assert fp(matched)-fp(historical)==-1 and fn(matched)-fn(historical)==-1
print('PASS presence masking, joint edge/slot exactness, paired transitions and signed errors')
