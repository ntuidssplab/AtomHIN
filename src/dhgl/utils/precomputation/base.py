from __future__ import annotations

from ...type import NType


def _generate_aliases(ntypes: list[NType]):
    aliases = {}
    used_aliases = set()

    for ntype in ntypes:
        alias = ''
        for i in range(1, len(ntype) + 1):
            alias = ntype[:i].capitalize()
            if alias not in used_aliases:
                break
        else:
            raise ValueError(f'Cannot generate unique alias for {ntype}')

        aliases[ntype] = alias
        used_aliases.add(alias)

    return aliases
