from typing import Any, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_models import Detection


def overlay_colored_masks(
    frame: np.ndarray,
    detections: List[Detection],
    colors: Any = (0, 0, 1),
) -> np.ndarray:
    """Color mask region for each detection on copied frame."""

    colors = np.array(px.colors.validate_colors(colors, 'tuple')) * 255

    frame = frame.copy()

    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]

        # Make masked regions grayscale and combine with color.
        x, y = detection.anchor
        h, w = detection.mask.shape
        frame[y : y + h, x : x + w][detection.mask] = (
            frame[y : y + h, x : x + w][detection.mask].mean(axis=-1)[..., None]
            + np.array(color)
        ) // 2

    return frame


def create_detections_figure(
    frame: np.ndarray,
    detections: List[Detection],
    mask_colors: Any = (0, 0, 1),
    box_colors: Any = (0, 1, 0),
    box_thickness: float = 1,
    annotation_colors: Any = (1, 1, 1),
    annotation_size: float = 14,
) -> go.Figure:
    """
    Create figure for visualizing detections on an image including:
    - colored masks
    - colored 2D bounding boxes including line thickness
    - colored id annotations including hover info

    To disable mask, boxes, or annotation, set the respective colors attribute to None.
    """

    # Colorize mask regions.
    if mask_colors is not None:
        frame = overlay_colored_masks(frame, detections, mask_colors)

    # Create figure from (colored) frame and remove excessive margin.
    fig = px.imshow(frame)
    fig.update_layout(margin=dict(r=16, l=16, t=16, b=16))

    # Create 2D bounding boxes.
    if box_colors is not None:
        box_colors = px.colors.validate_colors(box_colors, 'rgb')

        for i, detection in enumerate(detections):
            color = box_colors[i % len(box_colors)]

            x0, y0, x1, y1 = detection.box
            fig.add_shape(
                type='rect',
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line_color=color,
                line_width=box_thickness,
            )

    # Add annotations including hover info.
    if annotation_colors is not None:
        annotation_colors = px.colors.validate_colors(annotation_colors, 'rgb')

        xs = []
        ys = []
        texts = []
        colors = []
        customdata = []

        for i, detection in enumerate(detections):
            x, y = detection.anchor
            h, w = detection.mask.shape
            xs.append(x + w / 2)
            ys.append(y + h / 2)

            texts.append(str(i))
            colors.append(annotation_colors[i % len(annotation_colors)])
            customdata.append((i, detection.label, detection.score))

        fig.add_scatter(
            x=xs,
            y=ys,
            text=texts,
            mode='text',
            textfont=dict(color=colors, size=annotation_size),
            name='',
            customdata=customdata,
            hovertemplate='id: %{customdata[0]}<br>'
            + 'label: %{customdata[1]}<br>score: %{customdata[2]:.2f}',
        )

    return fig


def draw_path(
    fig: go.Figure,
    points: np.ndarray,
    color: Any = (0, 0, 0),
    thickness: float = 2,
    closed: bool = True,
):
    """Draw (closed) path on figure with points (2xn)."""

    color = px.colors.validate_colors(color, 'rgb')[0]

    pairs = [f'{x},{y}' for x, y in points.T]
    path = 'M ' + ' L '.join(pairs)
    if closed:
        path += ' Z'

    fig.add_shape(type='path', path=path, line_color=color, line_width=thickness)
