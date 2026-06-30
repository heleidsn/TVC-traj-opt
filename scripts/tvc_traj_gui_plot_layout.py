# -*- coding: utf-8 -*-
"""Responsive matplotlib layout helpers for the TVC GUI."""

from __future__ import annotations


def _disable_auto_layout(fig):
    """Turn off constrained/tight engines so GridSpec spacing is respected."""
    try:
        fig.set_layout_engine('none')
    except Exception:
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass


def setup_responsive_figure(fig, base_width_px=960, base_height_px=640, layout_mode='constrained'):
    """
    Configure figure for resize-aware reflow.

    layout_mode:
      - 'constrained': simple figures (single axes or tracking tabs)
      - 'gridspec': multi-panel figures built with GridSpec + subplots_adjust
    """
    fig._tvc_layout_mode = layout_mode  # noqa: SLF001
    fig._tvc_responsive_base = (float(base_width_px), float(base_height_px))  # noqa: SLF001

    if layout_mode == 'gridspec':
        _disable_auto_layout(fig)
        _apply_gridspec_subplots_adjust(fig, scale=1.0)
    else:
        try:
            fig.set_layout_engine('constrained')
        except Exception:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
        _set_constrained_pads(fig, scale=1.0)


def _apply_gridspec_subplots_adjust(fig, scale=1.0):
    """Apply manual margins for GridSpec-based overview / states / metrics tabs."""
    pad = getattr(fig, '_tvc_gridspec_pads', None) or {}
    hspace = float(pad.get('hspace', 0.32))
    wspace = float(pad.get('wspace', 0.26))
    tighten = 0.88 + 0.12 * float(scale)
    try:
        fig.subplots_adjust(
            left=float(pad.get('left', 0.06)),
            right=float(pad.get('right', 0.98)),
            top=float(pad.get('top', 0.92)),
            bottom=float(pad.get('bottom', 0.06)),
            hspace=hspace * tighten,
            wspace=wspace * tighten,
        )
    except Exception:
        pass


def _set_constrained_pads(fig, scale):
    pads = dict(
        w_pad=0.015,
        h_pad=0.015,
        hspace=max(0.04, 0.03 + 0.12 * (1.0 - scale)),
        wspace=max(0.04, 0.03 + 0.10 * (1.0 - scale)),
    )
    try:
        fig.set_constrained_layout_pads(**pads)
    except Exception:
        pass


def _scale_from_canvas(fig, canvas):
    base = getattr(fig, '_tvc_responsive_base', (960.0, 640.0))
    bw, bh = base
    w, h = canvas.get_width_height()
    if w <= 0 or h <= 0:
        return 1.0, w
    scale = min(w / bw, h / bh)
    return max(0.45, min(1.0, scale)), w


def place_legend(ax, canvas_width_px, n_items=6, handles=None, labels=None):
    """Place legend to reduce overlap on narrow panels."""
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if canvas_width_px < 420:
        ncol = min(3, max(1, n_items))
        fontsize = 5
        loc = 'upper center'
        bbox = (0.5, -0.28)
    elif canvas_width_px < 620:
        ncol = 1
        fontsize = 6
        loc = 'best'
        bbox = None
    else:
        ncol = 2 if n_items > 4 else 1
        fontsize = 7
        loc = 'best'
        bbox = None

    kwargs = dict(fontsize=fontsize, ncol=ncol, framealpha=0.85)
    if bbox is not None:
        kwargs.update(loc=loc, bbox_to_anchor=bbox, frameon=False)
    else:
        kwargs.update(loc=loc)
    ax.legend(handles, labels, **kwargs)


def _scale_axis_typography(ax, title_fs, label_fs, tick_fs, title_pad=2.0):
    if getattr(ax, 'name', None) == 'off':
        return
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=title_fs, pad=title_pad)
    xlabel = ax.get_xlabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs)
    ylabel = ax.get_ylabel()
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fs)
    zlabel = getattr(ax, 'get_zlabel', lambda: '')()
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    if hasattr(ax, 'tick_params'):
        try:
            ax.tick_params(axis='z', labelsize=tick_fs)
        except Exception:
            pass


def apply_responsive_layout(fig, canvas):
    """Scale titles, labels, ticks, and legends from canvas pixel size."""
    if fig is None or canvas is None:
        return
    scale, width_px = _scale_from_canvas(fig, canvas)
    layout_mode = getattr(fig, '_tvc_layout_mode', 'constrained')

    title_fs = max(7, int(9 * scale))
    label_fs = max(6, int(8 * scale))
    tick_fs = max(5, int(7 * scale))
    suptitle_fs = max(8, int(11 * scale))

    if layout_mode == 'gridspec':
        _apply_gridspec_subplots_adjust(fig, scale)
        for ax in fig.axes:
            _scale_axis_typography(ax, title_fs, label_fs, tick_fs)
        st = getattr(fig, '_suptitle', None)
        if st is not None:
            try:
                st.set_fontsize(suptitle_fs)
            except Exception:
                pass
    else:
        _set_constrained_pads(fig, scale)
        for ax in fig.axes:
            _scale_axis_typography(ax, title_fs, label_fs, tick_fs)
            if ax.get_legend_handles_labels()[0]:
                n_items = len(ax.get_legend_handles_labels()[0])
                place_legend(ax, width_px, n_items=n_items)
        st = getattr(fig, '_suptitle', None)
        if st is not None:
            try:
                st.set_fontsize(suptitle_fs)
            except Exception:
                pass

    try:
        canvas.draw_idle()
    except Exception:
        pass


def install_responsive_canvas(canvas, fig, base_width_px=960, base_height_px=640, layout_mode='constrained'):
    """Attach resize handler so plots reflow when the window is resized."""
    setup_responsive_figure(fig, base_width_px, base_height_px, layout_mode=layout_mode)

    def _on_resize(_event):
        apply_responsive_layout(fig, canvas)

    canvas.mpl_connect('resize_event', _on_resize)
    apply_responsive_layout(fig, canvas)
    return canvas
