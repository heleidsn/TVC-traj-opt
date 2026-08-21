# -*- coding: utf-8 -*-
"""Draw Bode / phase-margin panels for the Stability margins plot tab."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.stability_margins import (
    LOOP_ATTITUDE,
    LOOP_IDS,
    LOOP_POSITION,
    LOOP_RATE,
    LOOP_VELOCITY,
    format_margins_summary,
    format_margins_summary_all,
)

LOOP_TITLES = {
    LOOP_RATE: 'Rate',
    LOOP_ATTITUDE: 'Attitude',
    LOOP_VELOCITY: 'Velocity',
    LOOP_POSITION: 'Position',
}


def _style(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=9, fontweight='bold', pad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(True, which='both', alpha=0.35)


def _draw_one_bode(ax_mag, ax_phase, result: Dict[str, Any], *, show_xlabel: bool):
    """Draw mag/phase for one loop (solid=no lag, dashed=with actuator)."""
    title = LOOP_TITLES.get(result.get('loop'), str(result.get('loop', '')))
    w = np.asarray(result['w'], dtype=float)
    f_hz = w / (2.0 * np.pi)
    wo = result['without']
    wa = result['with_actuator']
    tau = float(result.get('tau_used_with_s', 0.0) or 0.0)

    if ax_mag is not None:
        _style(ax_mag, title, ylabel='|L| (dB)')
        ax_mag.semilogx(f_hz, wo['mag_db'], 'C0-', lw=1.4, label='no lag')
        ax_mag.semilogx(
            f_hz, wa['mag_db'], 'C3--', lw=1.3, alpha=0.9,
            label=f'τ={tau:.3f}s',
        )
        ax_mag.axhline(0.0, color='0.4', ls=':', lw=0.9)
        for key, color in (('without', 'C0'), ('with_actuator', 'C3')):
            w_gc = result[key].get('w_gc_rad_s')
            if w_gc is not None and np.isfinite(w_gc) and w_gc > 0:
                ax_mag.axvline(w_gc / (2 * np.pi), color=color, ls=':', lw=0.9, alpha=0.7)
        pm0 = wo.get('pm_deg', float('nan'))
        pm1 = wa.get('pm_deg', float('nan'))
        pm_txt = (
            f"PM {pm0:.0f}°→{pm1:.0f}°"
            if np.isfinite(pm0) and np.isfinite(pm1) else 'PM n/a'
        )
        ax_mag.text(
            0.02, 0.02, pm_txt, transform=ax_mag.transAxes,
            fontsize=7, va='bottom', ha='left', color='#333',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
        )

    if ax_phase is not None:
        _style(
            ax_phase,
            '',
            xlabel='Frequency (Hz)' if show_xlabel else '',
            ylabel='∠L (deg)',
        )
        ax_phase.semilogx(f_hz, wo['phase_deg'], 'C0-', lw=1.4)
        ax_phase.semilogx(f_hz, wa['phase_deg'], 'C3--', lw=1.3, alpha=0.9)
        ax_phase.axhline(-180.0, color='0.4', ls=':', lw=0.9)
        for key, color in (('without', 'C0'), ('with_actuator', 'C3')):
            blk = result[key]
            w_gc = blk.get('w_gc_rad_s')
            pm = blk.get('pm_deg')
            if w_gc is not None and np.isfinite(w_gc) and w_gc > 0 and np.isfinite(pm):
                f_gc = w_gc / (2 * np.pi)
                phase_at = -180.0 + pm
                ax_phase.plot([f_gc], [phase_at], 'o', color=color, ms=3.5)
                ax_phase.vlines(f_gc, -180.0, phase_at, colors=color, linestyles=':', lw=0.9)


def _apply_column_visibility(
    ax_by_loop: Dict[str, Any],
    visible: Dict[str, bool],
    gridspec=None,
):
    """Show/hide Bode columns and shrink hidden GridSpec width ratios."""
    ratios = []
    for loop in LOOP_IDS:
        on = bool(visible.get(loop, True))
        ratios.append(1.0 if on else 0.02)
        pair = ax_by_loop.get(loop) or {}
        for ax in (pair.get('ax_mag'), pair.get('ax_phase')):
            if ax is not None:
                ax.set_visible(on)
    ratios.append(0.95)  # info column
    if gridspec is None:
        return
    try:
        gridspec.set_width_ratios(ratios)
    except Exception:
        try:
            gridspec.update(width_ratios=ratios)
        except Exception:
            pass


def draw_stability_margins_panels(
    axes: Dict[str, Any],
    result: Optional[Dict[str, Any]],
    visible_loops: Optional[Dict[str, bool]] = None,
):
    """
    axes keys:
      * ax_by_loop: {loop_id: {'ax_mag', 'ax_phase'}, ...}  (preferred, all four)
      * or legacy: ax_mag, ax_phase, ax_info
      * ax_info: summary text panel
      * gridspec: optional GridSpec for column width updates
      * visible_loops: optional {loop_id: bool} (also accepted as arg)
    result: analyze_all_loops bundle, single analyze_loop dict, or None
    """
    ax_info = axes.get('ax_info')
    ax_by_loop = axes.get('ax_by_loop') or {}
    gridspec = axes.get('gridspec')
    visible = dict(visible_loops or axes.get('visible_loops') or {})
    for loop in LOOP_IDS:
        visible.setdefault(loop, True)

    # Clear all known axes
    for pair in ax_by_loop.values():
        for key in ('ax_mag', 'ax_phase'):
            ax = pair.get(key)
            if ax is not None:
                ax.clear()
    for key in ('ax_mag', 'ax_phase'):
        ax = axes.get(key)
        if ax is not None:
            ax.clear()
    if ax_info is not None:
        ax_info.clear()
        ax_info.axis('off')

    _apply_column_visibility(ax_by_loop, visible, gridspec)

    if result is None:
        placeholders = [
            (loop, pair) for loop, pair in ax_by_loop.items() if visible.get(loop, True)
        ]
        if not placeholders and axes.get('ax_mag') is not None:
            placeholders = [('rate', {'ax_mag': axes['ax_mag'], 'ax_phase': axes.get('ax_phase')})]
        for loop, pair in placeholders:
            title = LOOP_TITLES.get(loop, loop)
            for ax, ttl in (
                (pair.get('ax_mag'), title),
                (pair.get('ax_phase'), ''),
            ):
                if ax is None:
                    continue
                _style(ax, ttl)
                if ttl:
                    ax.text(
                        0.5, 0.5, 'Click “Update Bode”',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='#666',
                    )
        if ax_info is not None:
            ax_info.text(
                0.02, 0.98,
                'Stability margins (PX4 cascade)\n\n'
                'Toggle Rate / Attitude / Velocity / Position columns above.\n'
                'Solid = no lag · Dashed = with first-order actuator.\n'
                'Pitch/Roll→XY via attitude; Yaw: rate/att + Z thrust.\n'
                'Set gains / τ below, then click Update Bode.',
                ha='left', va='top', fontsize=8, family='monospace',
                transform=ax_info.transAxes,
            )
        return

    by_loop = result.get('by_loop')
    if isinstance(by_loop, dict) and ax_by_loop:
        drawn = 0
        for loop in LOOP_IDS:
            if not visible.get(loop, True):
                continue
            pair = ax_by_loop.get(loop) or {}
            one = by_loop.get(loop)
            if one is None:
                continue
            _draw_one_bode(
                pair.get('ax_mag'), pair.get('ax_phase'), one,
                show_xlabel=True,
            )
            if drawn > 0:
                for ax in (pair.get('ax_mag'), pair.get('ax_phase')):
                    if ax is not None:
                        ax.set_ylabel('')
            drawn += 1
    elif 'without' in result:
        # Legacy single-loop result
        _draw_one_bode(
            axes.get('ax_mag'), axes.get('ax_phase'), result, show_xlabel=True,
        )

    if ax_info is not None:
        if isinstance(by_loop, dict):
            summary = format_margins_summary_all(result, visible_loops=visible)
        else:
            summary = format_margins_summary(result)
        ax_info.text(
            0.02, 0.98, summary,
            ha='left', va='top', fontsize=8, family='monospace',
            transform=ax_info.transAxes,
        )
