"""Public codec mechanical tests, no neural training or GPU dependency."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import struct
import unittest

from tensegra.campaign02_memory import (
    BYTE_START, ROW_DIM, MemoryCapacityError, MemoryLimits,
    decode_memory, encode_memory, normalized_document,
)
from tensegra.campaign02_world import Workshop, generate_world, action_catalog, Action, Observation


class PublicMemoryTests(unittest.TestCase):
    def test_roundtrip_unknown_values_fields_and_float_bits(self):
        o=asdict(Workshop(generate_world(1)).observe())
        o["feedback"]={"status":"novel☀state", "free field": [None,True,False,-(2**63),2**63-1,-0.0,1e-9],
                       "empty":"", "ordered":{"b":2,"a":1}}
        actions=[{"kind":"think","arguments":{}}]
        memory=encode_memory(o,actions)
        decoded=decode_memory(memory)
        self.assertEqual(decoded,normalized_document(o,actions))
        self.assertEqual(struct.pack(">d",decoded["observation"]["feedback"]["free field"][5]),struct.pack(">d",-0.0))
        self.assertEqual(list(decoded["observation"]["feedback"]["ordered"]),["b","a"])
        self.assertEqual(len(memory.rows[0]),ROW_DIM)
        # Float32 carries exact byte / binary-index fractions without loss.
        rounded=deepcopy(memory)
        rounded.rows=[[struct.unpack('f',struct.pack('f',v))[0] for v in row] for row in memory.rows]
        self.assertEqual(decode_memory(rounded),decoded)

    def test_opaque_handle_renaming_preserves_all_features(self):
        env=Workshop(generate_world(3),address_seed=18)
        for row in env.observe().item_inventory[:3]:
            env.step(Action("inspect",{"target":row["handle"]}))
        env.step(Action("start_subset",{"handle":"draft-original"}))
        o=asdict(env.observe())
        actions=[asdict(a) for a in action_catalog(env.observe())]
        handles=[r["handle"] for r in o["item_inventory"]]+list(o["problems"])+[r["handle"] for r in o["records"]]
        mapping={h:f"renamed-{i*17+9}" for i,h in enumerate(handles)}
        # Include public fresh candidate handles; no inference about their values.
        for action in actions:
            if "handle" in action["arguments"]:
                h=action["arguments"]["handle"]
                mapping.setdefault(h,f"new-address-{len(mapping)}")
        def rename(x):
            if isinstance(x,dict):return {mapping.get(k,k):rename(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [rename(v) for v in x]
            return mapping.get(x,x) if isinstance(x,str) else x
        a,b=encode_memory(o,actions),encode_memory(rename(o),rename(actions))
        self.assertEqual(a.rows,b.rows)
        self.assertEqual(a.edges,b.edges)
        self.assertEqual(a.action_links,b.action_links)
        self.assertEqual(decode_memory(a),decode_memory(b))

    def test_historical_counterexample_memories_differ(self):
        path=Path(__file__).parents[1]/'research/campaigns/extended-02/diagnostics/a933-feature-collisions.json'
        raw=json.loads(path.read_text())
        for pair in raw['pairs'].values():
            a,b=(encode_memory(pair[s]['observation'],pair[s]['actions']) for s in ('left','right'))
            self.assertNotEqual(a.rows,b.rows)
            self.assertNotEqual(decode_memory(a),decode_memory(b))

    def test_unknown_mask_order_and_candidate_links(self):
        o=asdict(Workshop(generate_world(2)).observe())
        actions=[{"kind":"inspect","arguments":{"target":o['item_inventory'][0]['handle']}},
                 {"kind":"inspect","arguments":{"target":o['item_inventory'][1]['handle']}}]
        a=encode_memory(o,actions)
        self.assertEqual(len(a.action_links),2)
        self.assertNotEqual(a.action_links[0],a.action_links[1])
        self.assertTrue(all(root in links for root,links in zip(a.action_roots,a.action_links)))
        self.assertTrue(any(node<a.action_roots[0] for node in a.action_links[0]))
        observed=deepcopy(o); observed['known_edges']=[]
        self.assertNotEqual(a.rows,encode_memory(observed,actions).rows)
        self.assertIsNone(decode_memory(a)['observation']['known_edges'])
        self.assertEqual(decode_memory(encode_memory(observed,actions))['observation']['known_edges'],[])
        self.assertNotEqual(a.rows,encode_memory(o,list(reversed(actions))).rows)

    def test_new_problem_keeps_existing_record_identity_and_fresh_binding(self):
        env=Workshop(generate_world(41),address_seed=22)
        env.step(Action("inspect",{"target":env.observe().item_inventory[0]["handle"]}))
        before=env.observe()
        actions=action_catalog(before)
        draft=next(a for a in actions if a.kind=="start_subset")
        old=decode_memory(encode_memory(before,actions))
        old_return=old["observation"]["records"][0]["handle"]
        future=old["actions"][actions.index(draft)]["arguments"]["handle"]
        env.step(draft)
        new=decode_memory(encode_memory(env.observe()))
        self.assertEqual(old_return,new["observation"]["records"][0]["handle"])
        self.assertIn(f"@handle:{future['@opaque']}",new["observation"]["problems"])
        self.assertNotEqual(old_return,future)

    def test_typed_namespace_collision_and_literal_map(self):
        o=asdict(Workshop(generate_world(1,categories=1,choices=1)).observe())
        o["item_inventory"]=[{"handle":"same","category":0}]
        o["known_items"]={"same":{"handle":"same","category":0,"weight":1,"price":1}}
        o["problems"]={"same":{"primitive":"constrained_subset","problem":{"handles":["same"]},"state_version":0}}
        o["records"]=[{"handle":"same","kind":"computation","problem":"same","source":"same","state_version":0}]
        actions=[{"kind":"choose_item","arguments":{"item":"same"}},
                 {"kind":"call","arguments":{"problem":"same","budget":16}},
                 {"kind":"retrieve","arguments":{"handle":"same"}},
                 {"kind":"inspect","arguments":{"target":"map"}}]
        decoded=decode_memory(encode_memory(o,actions))
        bindings=[decoded["actions"][i]["arguments"][key]["@opaque"]
                  for i,key in enumerate(("item","problem","handle"))]
        self.assertEqual(bindings,[0,512,1024])
        self.assertEqual(decoded["actions"][3]["arguments"]["target"],"map")
        self.assertEqual(decoded,normalized_document(o,actions))
        o["item_inventory"]=[{"handle":"map","category":0}]
        with self.assertRaises(ValueError):
            encode_memory(o,actions)

    def test_capacity_never_silently_truncates(self):
        o=Workshop(generate_world(1)).observe()
        with self.assertRaises(MemoryCapacityError) as cm:
            encode_memory(o,limits=MemoryLimits(max_nodes=10))
        self.assertEqual(cm.exception.limit,10)
        self.assertEqual(cm.exception.observed,11)
        self.assertEqual(cm.exception.record()['status'],'unsupported_public_memory')
        with self.assertRaises(MemoryCapacityError):
            encode_memory(o,limits=MemoryLimits(max_handles=1))
        with self.assertRaises(MemoryCapacityError):
            encode_memory({'feedback':{'x':'hello'}},[],MemoryLimits(max_string_bytes=2))
        with self.assertRaises(ValueError):
            encode_memory({'feedback':{'x':float('nan')}},[])
        with self.assertRaises(MemoryCapacityError):
            encode_memory({'feedback':{'x':2**64}},[])


if __name__=='__main__':
    unittest.main()
