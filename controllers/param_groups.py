# -*- coding: utf-8 -*-
"""Grouped controller parameter schemas for the Tracking GUI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .px4_params import px4_param_groups


def _lqr_param_groups() -> List[Dict[str, Any]]:
    from .params import _LQR_SPECS
    by_key = {s['key']: s for s in _LQR_SPECS}
    return [
        {
            'title': 'State weight Q — position',
            'specs': [by_key[k] for k in ('Q_x', 'Q_y', 'Q_z')],
        },
        {
            'title': 'State weight Q — velocity',
            'specs': [by_key[k] for k in ('Q_vx', 'Q_vy', 'Q_vz')],
        },
        {
            'title': 'State weight Q — attitude',
            'specs': [by_key[k] for k in ('Q_qx', 'Q_qy', 'Q_qz')],
        },
        {
            'title': 'State weight Q — angular rate',
            'specs': [by_key[k] for k in ('Q_p', 'Q_q', 'Q_r')],
        },
        {
            'title': 'Control weight R',
            'specs': [by_key[k] for k in ('R_qx', 'R_qy', 'R_T', 'R_r')],
        },
    ]


def _nmpc_param_groups() -> List[Dict[str, Any]]:
    from .params import _NMPC_SPECS
    by_key = {s['key']: s for s in _NMPC_SPECS}
    groups = [
        {
            'title': 'NMPC settings',
            'specs': [
                by_key['horizon'], by_key['nmpc_dt'],
                by_key['nmpc_du_weight'], by_key['nmpc_terminal_scale'],
                by_key['nmpc_nlp_max_iter'],
            ],
        },
    ]
    groups.extend(_lqr_param_groups())
    return groups


def _mpc_param_groups() -> List[Dict[str, Any]]:
    from .params import _MPC_SPECS
    by_key = {s['key']: s for s in _MPC_SPECS}
    groups = [
        {
            'title': 'MPC settings',
            'specs': [by_key['horizon'], by_key['mpc_dt']],
        },
    ]
    groups.extend(_lqr_param_groups())
    return groups


def param_groups_for(
    controller_id: str,
    px4_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Return categorized parameter groups for the Tracking GUI."""
    from .params import (
        CONTROLLER_ACADOS_NMPC,
        CONTROLLER_LQR,
        CONTROLLER_MPC,
        CONTROLLER_PX4,
    )
    cid = controller_id if controller_id in (
        CONTROLLER_PX4, CONTROLLER_LQR, CONTROLLER_MPC, CONTROLLER_ACADOS_NMPC,
    ) else CONTROLLER_PX4
    groups: List[Dict[str, Any]] = []
    if cid == CONTROLLER_PX4:
        share = True
        if px4_params is not None:
            share = bool(px4_params.get('share_rp_gains', True))
        groups = px4_param_groups(share_rp=share)
    elif cid == CONTROLLER_LQR:
        groups = _lqr_param_groups()
    elif cid == CONTROLLER_MPC:
        groups = _mpc_param_groups()
    elif cid == CONTROLLER_ACADOS_NMPC:
        groups = _nmpc_param_groups()
    return deepcopy(groups)


def flatten_param_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for group in groups:
        specs.extend(group.get('specs') or [])
    return specs
