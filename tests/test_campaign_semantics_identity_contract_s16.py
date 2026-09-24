"""S16 public-only normalization mechanics; no model or graph targets."""
import re
import pytest
import torch
from topoformer.thinking_language import ActorInput
from topoformer.semantic_curriculum import encode_text
from topoformer.campaign_semantics_identity_contract import (
    canonicalize_public_identifiers,restore_public_copy_tokens)


def public(arity=3):
    names=['frank','bob','alice','erin','carol'][:arity]
    pattern=names.copy();pattern[1]='D'
    def parent(args):
        return args[0]+' parent '+args[1] if len(args)==2 else 'parent: '+', '.join(args[:-1])+' and '+args[-1]
    return ActorInput('You know these facts: '+parent(names)+'; '+parent(list(reversed(names)))+'.\nThe pattern is '+parent(pattern)+'.\nWhat is the unify D?',())


@pytest.mark.parametrize('arity',[2,3,4,5])
def test_bijective_rename_gives_identical_complete_public_features(arity):
    original=public(arity)
    rename={'frank':'nuvfz','bob':'nuvbz','alice':'nuvaz','erin':'nuvez','carol':'nuvcz','D':'NUVDZ'}
    renamed=ActorInput(re.sub(r'\w+',lambda m:rename.get(m.group(),m.group()),original.text),())
    a=canonicalize_public_identifiers(original);b=canonicalize_public_identifiers(renamed)
    assert a.public==b.public
    assert a.arity==arity
    af,al=encode_text(a.public.text);bf,bl=encode_text(b.public.text)
    assert al==bl and torch.equal(af,bf)
    assert a.identity_positions==b.identity_positions
    for i in range(len(a.original_tokens)):
        assert a.original_tokens.index(a.original_tokens[i])==a.canonical_tokens.index(a.canonical_tokens[i])
        if i not in a.identity_positions:assert a.original_tokens[i]==a.canonical_tokens[i]
    assert restore_public_copy_tokens(a,range(len(a.original_tokens)))==a.original_tokens
    assert restore_public_copy_tokens(b,range(len(b.original_tokens)))==b.original_tokens


def test_simultaneous_alias_swaps_idempotence_and_no_wrong_copy_repair():
    a=canonicalize_public_identifiers(public())
    assert dict(a.original_to_canonical)['frank']=='alice'
    assert dict(a.original_to_canonical)['alice']=='carol'
    assert canonicalize_public_identifiers(a.public).public==a.public
    assert restore_public_copy_tokens(a,[0,-1])==('You',None)
    with pytest.raises(ValueError):restore_public_copy_tokens(a,[len(a.original_tokens)])
    with pytest.raises(TypeError):restore_public_copy_tokens(a,[1.5])


def test_equality_change_and_argument_order_not_erased():
    a=public()
    merged=ActorInput(a.text.replace('alice','bob'),())
    assert canonicalize_public_identifiers(a).public!=canonicalize_public_identifiers(merged).public
    # Only one occurrence is moved, changing a role relation rather than globally renaming.
    changed=ActorInput(a.text.replace('frank, D and alice','alice, D and frank'),())
    assert canonicalize_public_identifiers(a).public!=canonicalize_public_identifiers(changed).public


@pytest.mark.parametrize('text',[
    'Unrecognized arbitrary English alice.',
    'You know these facts: alice parent bob. The pattern is Alice parent bob. What is the unify A?',
    'You know these facts: alice parent bob. The pattern is A parent parent. What is the unify A?',
    'You know these facts: parent: a, b, c, d, e and f. The pattern is a parent A. What is the unify A?',
    'You know these facts: a parent b; c parent d; e parent f; g parent a. The pattern is a parent A. What is the unify A?',
])
def test_unsupported_or_overcapacity_public_inputs_rejected(text):
    with pytest.raises(ValueError):canonicalize_public_identifiers(ActorInput(text,()))


def test_no_metadata_or_options_accepted():
    with pytest.raises(ValueError):canonicalize_public_identifiers({'text':public().text,'nodes':['hidden']})
    with pytest.raises(ValueError):canonicalize_public_identifiers(ActorInput(public().text,('private choice',)))


def test_visible_case_namespaces_and_reserved_word_resemblance():
    text='You know these facts: a parent parenthood. The pattern is A parent parenthood. What is the unify A?'
    result=canonicalize_public_identifiers(ActorInput(text,()))
    mapping=dict(result.original_to_canonical)
    assert mapping=={'a':'alice','parenthood':'bob','A':'A'}
    assert result.public.text.count(' parent ')==2
    assert restore_public_copy_tokens(result,[result.original_tokens.index('parent')])==('parent',)
