"""C02 public identity alpha-renaming; no target or return tensors are inputs."""
import torch

REFERENCES = ('instruction_destinations', 'instruction_arguments', 'query_destination')


def identity_links(public):
    """Recover exact public equality links, without locating the requested record."""
    keys = public['keys']
    if keys.ndim != 3 or keys.shape[-1] != 32:
        raise ValueError('public keys must have shape batch,candidates,32')
    same = (keys[:, :, None] == keys[:, None, :]).all(-1)
    if not (same.sum(-1) == 1).all():
        raise ValueError('candidate identities must be unique')
    null = (keys == 0).all(-1)
    if not (null.sum(-1) == 1).all():
        raise ValueError('exactly one zero null identity required')
    links = {}
    for name in REFERENCES:
        ref = public[name]
        flat = ref.reshape(len(keys), -1, 32)
        matches = (flat[:, :, None] == keys[:, None]).all(-1)
        if not (matches.sum(-1) == 1).all():
            raise ValueError(f'{name} must refer to unique public candidate identities')
        links[name] = matches.long().argmax(-1).reshape(ref.shape[:-1])
    return links


def rekey(public, generator):
    """Fresh Gaussian codes per row/presentation; preserve null and all other data.

    The dedicated CPU generator never consumes model or sampling RNG. Candidate
    indices and every target remain unchanged; repeated base rows get new views.
    """
    links = identity_links(public)
    keys = public['keys']
    codes = torch.randn(keys.shape, generator=generator, dtype=keys.dtype, device='cpu').to(keys.device)
    codes = codes.masked_fill((keys == 0).all(-1, keepdim=True), 0)
    # Reject the negligible floating-point collision case, never merge identities.
    if not (((codes[:, :, None] == codes[:, None, :]).all(-1)).sum(-1) == 1).all():
        raise RuntimeError('new public identity collision')
    result = dict(public, keys=codes)
    for name, indices in links.items():
        result[name] = codes.gather(1, indices.reshape(len(keys), -1, 1).expand(-1, -1, 32)).reshape(public[name].shape)
    return result
